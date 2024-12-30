import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import random
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# XGBoost
from xgboost import XGBClassifier

# For capturing convergence warnings
from sklearn.exceptions import ConvergenceWarning
from tkinter import ttk

class ClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning Classifier Comparison Tool")

        # Data holders
        self.data = None
        self.class_column = None
        self.results = []

        # Main Notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab: File loading and dataset info
        self.file_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.file_frame, text="Data")

        # Tab: Classifier selection
        self.classifiers_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.classifiers_frame, text="Classifiers")

        # Tab: Hyperparameters
        self.params_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.params_tab, text="Parameters")

        # Tab: Results
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Results")

        # Build each section
        self.build_file_tab()
        self.build_classifiers_tab()
        self.build_params_tab()
        self.build_results_tab()

    # ------------------------------------------------------------------------
    # Tab 1: File loading and info
    # ------------------------------------------------------------------------
    def build_file_tab(self):
        # File label and load button
        self.file_label = tk.Label(self.file_frame, text="No file loaded")
        self.file_label.pack(pady=5)

        self.load_button = tk.Button(self.file_frame, text="Load File", command=self.load_file)
        self.load_button.pack(pady=5)

        # Info text
        self.info_text = tk.Text(self.file_frame, height=10, state=tk.DISABLED, wrap=tk.WORD)
        self.info_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            self.data = pd.read_csv(file_path)
            self.class_column = self.find_class_column()
            self.display_dataset_info()
            self.file_label.config(text=f"Loaded: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def find_class_column(self):
        for col in self.data.columns:
            if "class" in col.lower():
                return col
        raise ValueError("No column with 'class' found.")

    def display_dataset_info(self):
        class_counts = self.data[self.class_column].value_counts()
        total_cases = len(self.data)
        num_classes = len(class_counts)
        balance_ratio = class_counts.min() / class_counts.max() if class_counts.max() != 0 else 0

        info = (
            f"Classes: {list(class_counts.index)}\n"
            f"Class Counts: {class_counts.to_dict()}\n"
            f"Total Cases: {total_cases}\n"
            f"Number of Classes: {num_classes}\n"
            f"Class Balance: {'Balanced' if abs(balance_ratio - 1.0) < 1e-9 else 'Not Balanced'} "
            f"(Ratio: {balance_ratio:.2f})\n"
            f"Number of Features: {len(self.data.columns) - 1}\n"  # -1 to exclude class column
            f"Features: {[col for col in self.data.columns if col != self.class_column]}\n"
            f"Missing Values: "
            f"{'none' if self.data.isnull().sum().sum() == 0 else f'contains missing values ({self.data.isnull().sum().sum()})'}"
        )

        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info)
        self.info_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------------
    # Tab 2: Classifier selection
    # ------------------------------------------------------------------------
    def build_classifiers_tab(self):
        self.selected_classifiers = {}

        # We'll place them in two rows, each with 7 classifiers
        top_frame = ttk.Frame(self.classifiers_frame)
        top_frame.pack(fill=tk.X, pady=5)

        bottom_frame = ttk.Frame(self.classifiers_frame)
        bottom_frame.pack(fill=tk.X, pady=5)

        # Classifiers in the order requested:
        classifiers_row1 = [
            "Decision Tree",
            "Random Forest",
            "Extra Trees",
            "KNN",
            "SVM",
            "LDA",
            "Logistic Regression",
        ]
        classifiers_row2 = [
            "Ridge",
            "Naive Bayes",
            "MLP",
            "SGD",
            "Gradient Boosting",
            "AdaBoost",
            "XGBoost",
        ]

        # Add row1 checkboxes
        for i, clf_name in enumerate(classifiers_row1):
            var = tk.BooleanVar(value=False)
            chk = tk.Checkbutton(top_frame, text=clf_name, variable=var)
            chk.grid(row=0, column=i, padx=5, pady=5, sticky=tk.W)
            self.selected_classifiers[clf_name] = var

        # Add row2 checkboxes
        for i, clf_name in enumerate(classifiers_row2):
            var = tk.BooleanVar(value=False)
            chk = tk.Checkbutton(bottom_frame, text=clf_name, variable=var)
            chk.grid(row=1, column=i, padx=5, pady=5, sticky=tk.W)
            self.selected_classifiers[clf_name] = var

        # Common training params
        common_params_frame = ttk.LabelFrame(self.classifiers_frame, text="Common CV Parameters")
        common_params_frame.pack(fill=tk.X, pady=10, padx=5)

        self.cross_val_split = self.add_param_entry(
            parent=common_params_frame,
            label="Cross-Validation Split (>=2):",
            default="5"
        )
        self.run_count = self.add_param_entry(
            parent=common_params_frame,
            label="Run Count (>=1):",
            default="10"
        )
        self.random_seed = self.add_param_entry(
            parent=common_params_frame,
            label="Random Seed:",
            default=str(random.randint(0, 10**6))
        )

        # Button to run
        self.run_button = tk.Button(
            self.classifiers_frame,
            text="Run Selected Classifiers",
            command=self.run_classifiers
        )
        self.run_button.pack(pady=5)

    # Helper for creating labeled entries (for cross_val_split, run_count, random_seed)
    def add_param_entry(self, parent, label, default):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)

        lbl = ttk.Label(frame, text=label)
        lbl.pack(side=tk.LEFT, padx=5)

        entry = ttk.Entry(frame, width=10)
        entry.insert(0, default)
        entry.pack(side=tk.LEFT, padx=5)

        return entry

    # ------------------------------------------------------------------------
    # Tab 3: Hyperparameter controls
    # ------------------------------------------------------------------------
    def build_params_tab(self):
        """
        Creates a scrollable frame with hyperparameter inputs.
        For non-numeric discrete choices, we provide a dropdown (Combobox).
        For numeric inputs, we provide an Entry plus a short label indicating suggested range/meaning.
        """
        # Create a Canvas + Scrollbar for the hyperparam entries
        canvas = tk.Canvas(self.params_tab)
        scrollbar = ttk.Scrollbar(self.params_tab, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Parameter definitions: type, default, help, [options] for combos
        self.param_config = {
            "Decision Tree": {
                "criterion": {
                    "type": "combo",
                    "options": ["gini", "entropy", "log_loss"],
                    "default": "gini",
                    "help": "Split quality function"
                },
                "max_depth": {
                    "type": "numeric",
                    "default": "None",
                    "help": "Max depth (None=unbounded)"
                }
            },
            "Random Forest": {
                "n_estimators": {
                    "type": "numeric",
                    "default": "100",
                    "help": "Number of trees (e.g. 50-300)"
                },
                "criterion": {
                    "type": "combo",
                    "options": ["gini", "entropy", "log_loss"],
                    "default": "gini",
                    "help": "Split quality function"
                },
                "max_depth": {
                    "type": "numeric",
                    "default": "None",
                    "help": "Max depth (None=unbounded)"
                }
            },
            "Extra Trees": {
                "n_estimators": {
                    "type": "numeric",
                    "default": "100",
                    "help": "Number of trees (e.g. 50-300)"
                },
                "criterion": {
                    "type": "combo",
                    "options": ["gini", "entropy", "log_loss"],
                    "default": "gini",
                    "help": "Split quality function"
                },
                "max_depth": {
                    "type": "numeric",
                    "default": "None",
                    "help": "Max depth (None=unbounded)"
                }
            },
            "KNN": {
                "n_neighbors": {
                    "type": "numeric",
                    "default": "5",
                    "help": "Number of neighbors (1-20 typical)"
                },
                "weights": {
                    "type": "combo",
                    "options": ["uniform", "distance"],
                    "default": "uniform",
                    "help": "Uniform or distance-based weighting"
                }
            },
            "SVM": {
                "kernel": {
                    "type": "combo",
                    "options": ["linear", "rbf", "poly", "sigmoid"],
                    "default": "rbf",
                    "help": "SVM kernel type"
                },
                "C": {
                    "type": "numeric",
                    "default": "1.0",
                    "help": "Regularization strength (0.001-100)"
                }
            },
            "LDA": {
                "solver": {
                    "type": "combo",
                    "options": ["svd", "lsqr", "eigen"],
                    "default": "svd",
                    "help": "LDA solver"
                }
            },
            "Logistic Regression": {
                "solver": {
                    "type": "combo",
                    "options": ["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
                    "default": "lbfgs",
                    "help": "Solver for optimization"
                },
                "max_iter": {
                    "type": "numeric",
                    "default": "1000",
                    "help": "Max iteration count"
                },
                "C": {
                    "type": "numeric",
                    "default": "1.0",
                    "help": "Inverse reg. strength (0.001-100)"
                }
            },
            "Ridge": {
                "alpha": {
                    "type": "numeric",
                    "default": "1.0",
                    "help": "Regularization strength (0.0+)"
                }
            },
            "Naive Bayes": {
                "var_smoothing": {
                    "type": "numeric",
                    "default": "1e-9",
                    "help": "Stability var. (1e-12 ~ 1e-7 typical)"
                }
            },
            "MLP": {
                "hidden_layer_sizes": {
                    "type": "text",
                    "default": "(100, )",
                    "help": "Layer sizes tuple, e.g. (100,50)"
                },
                "activation": {
                    "type": "combo",
                    "options": ["identity", "logistic", "tanh", "relu"],
                    "default": "relu",
                    "help": "Activation function"
                },
                "solver": {
                    "type": "combo",
                    "options": ["lbfgs", "sgd", "adam"],
                    "default": "adam",
                    "help": "MLP solver"
                },
                "alpha": {
                    "type": "numeric",
                    "default": "0.0001",
                    "help": "L2 penalty parameter"
                },
                # Expose max_iter in the GUI
                "max_iter": {
                    "type": "numeric",
                    "default": "200",
                    "help": "Max training iterations"
                }
            },
            "SGD": {
                "loss": {
                    "type": "combo",
                    "options": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
                    "default": "hinge",
                    "help": "Loss function"
                },
                "penalty": {
                    "type": "combo",
                    "options": ["l2", "l1", "elasticnet"],
                    "default": "l2",
                    "help": "Regularization type"
                },
                "alpha": {
                    "type": "numeric",
                    "default": "0.0001",
                    "help": "Constant for regularization"
                }
            },
            "Gradient Boosting": {
                "n_estimators": {
                    "type": "numeric",
                    "default": "100",
                    "help": "Number of boosting stages"
                },
                "learning_rate": {
                    "type": "numeric",
                    "default": "0.1",
                    "help": "Learning rate (0.01-1.0 typical)"
                },
                "max_depth": {
                    "type": "numeric",
                    "default": "3",
                    "help": "Max depth (1-10 typical)"
                }
            },
            "AdaBoost": {
                "n_estimators": {
                    "type": "numeric",
                    "default": "100",
                    "help": "Number of boosting stages"
                },
                "learning_rate": {
                    "type": "numeric",
                    "default": "1.0",
                    "help": "Weight applied to each classifier"
                }
            },
            "XGBoost": {
                "n_estimators": {
                    "type": "numeric",
                    "default": "100",
                    "help": "Number of boosting rounds"
                },
                "learning_rate": {
                    "type": "numeric",
                    "default": "0.1",
                    "help": "Step size shrinkage (0.01-0.3 typical)"
                },
                "max_depth": {
                    "type": "numeric",
                    "default": "3",
                    "help": "Max tree depth (1-10 typical)"
                }
            }
        }

        self.hyperparam_entries = {}
        row_idx = 0

        for clf_name, param_dict in self.param_config.items():
            group = ttk.LabelFrame(scroll_frame, text=clf_name)
            group.pack(fill=tk.X, padx=5, pady=5)

            self.hyperparam_entries[clf_name] = {}
            for param_name, info in param_dict.items():
                param_type = info["type"]
                default_value = info["default"]
                param_help = info["help"]

                row_frame = ttk.Frame(group)
                row_frame.pack(fill=tk.X, pady=2)

                lbl = ttk.Label(row_frame, text=param_name + ":")
                lbl.pack(side=tk.LEFT, padx=5)

                if param_type == "combo":
                    combo = ttk.Combobox(row_frame, values=info["options"], state="readonly", width=15)
                    combo.set(default_value)
                    combo.pack(side=tk.LEFT, padx=5)
                    self.hyperparam_entries[clf_name][param_name] = combo

                elif param_type == "numeric":
                    entry = ttk.Entry(row_frame, width=10)
                    entry.insert(0, str(default_value))
                    entry.pack(side=tk.LEFT, padx=5)
                    self.hyperparam_entries[clf_name][param_name] = entry

                elif param_type == "text":
                    entry = ttk.Entry(row_frame, width=15)
                    entry.insert(0, str(default_value))
                    entry.pack(side=tk.LEFT, padx=5)
                    self.hyperparam_entries[clf_name][param_name] = entry

                # Add small help text / range note
                help_label = ttk.Label(row_frame, text=f"({param_help})", foreground="gray")
                help_label.pack(side=tk.LEFT, padx=5)

            row_idx += 1

    # ------------------------------------------------------------------------
    # Tab 4: Results
    # ------------------------------------------------------------------------
    def build_results_tab(self):
        # Table for results
        self.results_table = ttk.Treeview(
            self.results_tab,
            columns=("Classifier", "Best", "Worst", "Average", "Std Dev"),
            show="headings",
            height=10
        )
        for col in self.results_table['columns']:
            self.results_table.heading(col, text=col)
            self.results_table.column(col, width=100, anchor=tk.CENTER)
        self.results_table.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)

        # Export button
        self.export_button = tk.Button(
            self.results_tab,
            text="Export to CSV",
            command=self.export_results
        )
        self.export_button.pack(pady=5)

    # ------------------------------------------------------------------------
    # Running classifiers
    # ------------------------------------------------------------------------
    def run_classifiers(self):
        if self.data is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return

        try:
            num_splits = int(self.cross_val_split.get())
            if num_splits < 2:
                raise ValueError("Cross-validation splits must be at least 2")

            run_count = int(self.run_count.get())
            if run_count < 1:
                raise ValueError("Run count must be at least 1")

            random_seed = int(self.random_seed.get())

            # Prepare feature/target
            X = self.data.drop(columns=self.class_column)
            y = self.data[self.class_column]

            # ---------------------------------------------------------------------
            # AUTOMATIC LABEL ENCODING IF CLASSES ARE STRINGS
            # ---------------------------------------------------------------------
            if y.dtype == object or y.dtype.kind in {'U', 'S'}:
                le = LabelEncoder()
                y = le.fit_transform(y)

            self.results = []
            any_convergence_issue = False

            # For each selected classifier, parse hyperparams and run
            for clf_name, var in self.selected_classifiers.items():
                if not var.get():
                    continue

                # Build the classifier with the hyperparams from UI
                hyperparams = self.parse_hyperparams(clf_name)
                classifier = self.build_classifier(clf_name, hyperparams, random_seed)

                accuracies = []
                # Catch warnings about convergence
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", ConvergenceWarning)

                    for run in range(run_count):
                        seed_for_run = random_seed + run
                        random.seed(seed_for_run)
                        np.random.seed(seed_for_run)

                        cv = KFold(n_splits=num_splits, shuffle=True, random_state=seed_for_run)
                        scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
                        accuracies.extend(scores)

                    # Check if any ConvergenceWarning was raised
                    for warning_msg in w:
                        if issubclass(warning_msg.category, ConvergenceWarning):
                            any_convergence_issue = True

                best = max(accuracies)
                worst = min(accuracies)
                average = np.mean(accuracies)
                std_dev = np.std(accuracies)

                self.results.append((clf_name, best, worst, average, std_dev))

            # If at least one ConvergenceWarning was encountered, show a friendly info message
            if any_convergence_issue:
                messagebox.showinfo(
                    "Convergence Notice",
                    "Some classifiers reached max_iter before converging. "
                    "Consider increasing max_iter or adjusting other hyperparameters."
                )

            self.update_results_table()
            self.notebook.select(self.results_tab)

        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def parse_hyperparams(self, clf_name):
        """
        Reads the hyperparameter widgets for the given classifier name.
        """
        param_values = {}
        config = self.param_config[clf_name]
        for param_name, widget in self.hyperparam_entries[clf_name].items():
            param_info = config[param_name]
            param_type = param_info["type"]
            raw_val = widget.get().strip()

            if param_type == "combo":
                # Directly use selected string
                param_values[param_name] = raw_val
            elif param_type in ["numeric", "text"]:
                if raw_val.lower() == "none":
                    param_values[param_name] = None
                else:
                    # Try integer
                    try:
                        param_values[param_name] = int(raw_val)
                        continue
                    except ValueError:
                        pass
                    # Try float
                    try:
                        param_values[param_name] = float(raw_val)
                        continue
                    except ValueError:
                        pass
                    # If parentheses => treat as eval (for tuples)
                    if raw_val.startswith("(") or raw_val.startswith("["):
                        try:
                            param_values[param_name] = eval(raw_val)
                            continue
                        except:
                            pass
                    # Otherwise, store string
                    param_values[param_name] = raw_val

        return param_values

    def build_classifier(self, clf_name, hyperparams, random_seed):
        """
        Instantiate the classifier with the parsed hyperparameters.
        We also pass in 'random_state' if relevant to that classifier.
        """
        if clf_name == "Decision Tree":
            return DecisionTreeClassifier(
                random_state=random_seed,
                **{k: v for k, v in hyperparams.items() if k in ["criterion", "max_depth"]}
            )
        elif clf_name == "Random Forest":
            return RandomForestClassifier(
                random_state=random_seed,
                **{k: v for k, v in hyperparams.items() if k in ["n_estimators", "criterion", "max_depth"]}
            )
        elif clf_name == "Extra Trees":
            return ExtraTreesClassifier(
                random_state=random_seed,
                **{k: v for k, v in hyperparams.items() if k in ["n_estimators", "criterion", "max_depth"]}
            )
        elif clf_name == "KNN":
            return KNeighborsClassifier(
                **{k: v for k, v in hyperparams.items() if k in ["n_neighbors", "weights"]}
            )
        elif clf_name == "SVM":
            return SVC(
                random_state=random_seed,
                **{k: v for k, v in hyperparams.items() if k in ["kernel", "C"]}
            )
        elif clf_name == "LDA":
            return LinearDiscriminantAnalysis(
                **{k: v for k, v in hyperparams.items() if k in ["solver"]}
            )
        elif clf_name == "Logistic Regression":
            return LogisticRegression(
                random_state=random_seed,
                **{k: v for k, v in hyperparams.items() if k in ["solver", "max_iter", "C"]}
            )
        elif clf_name == "Ridge":
            return RidgeClassifier(
                random_state=random_seed,
                **{k: v for k, v in hyperparams.items() if k in ["alpha"]}
            )
        elif clf_name == "Naive Bayes":
            return GaussianNB(
                **{k: v for k, v in hyperparams.items() if k in ["var_smoothing"]}
            )
        elif clf_name == "MLP":
            return MLPClassifier(
                random_state=random_seed,
                **{
                    k: v for k, v in hyperparams.items()
                    if k in ["hidden_layer_sizes", "activation", "solver", "alpha", "max_iter"]
                }
            )
        elif clf_name == "SGD":
            return SGDClassifier(
                random_state=random_seed,
                **{k: v for k, v in hyperparams.items() if k in ["loss", "penalty", "alpha"]}
            )
        elif clf_name == "Gradient Boosting":
            return GradientBoostingClassifier(
                random_state=random_seed,
                **{k: v for k, v in hyperparams.items() if k in ["n_estimators", "learning_rate", "max_depth"]}
            )
        elif clf_name == "AdaBoost":
            return AdaBoostClassifier(
                random_state=random_seed,
                **{k: v for k, v in hyperparams.items() if k in ["n_estimators", "learning_rate"]}
            )
        elif clf_name == "XGBoost":
            return XGBClassifier(
                random_state=random_seed,
                use_label_encoder=False,
                eval_metric='logloss',
                **{k: v for k, v in hyperparams.items() if k in ["n_estimators", "learning_rate", "max_depth"]}
            )
        else:
            raise ValueError(f"Unknown classifier: {clf_name}")

    # ------------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------------
    def update_results_table(self):
        for row in self.results_table.get_children():
            self.results_table.delete(row)

        for result in self.results:
            clf_name, best, worst, average, std_dev = result
            formatted_result = (
                clf_name,
                f"{best:.4f}",
                f"{worst:.4f}",
                f"{average:.4f}",
                f"{std_dev:.4f}"
            )
            self.results_table.insert("", "end", values=formatted_result)

    def export_results(self):
        if not self.results:
            messagebox.showerror("Error", "No results to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv", 
            filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            return

        df = pd.DataFrame(self.results, columns=["Classifier", "Best", "Worst", "Average", "Std Dev"])
        df.to_csv(file_path, index=False)
        messagebox.showinfo("Success", "Results exported successfully.")


if __name__ == "__main__":
    root = tk.Tk()

    # Define a small utility function to center the window.
    def center_window(win):
        win.update_idletasks()   # Make sure we have the correct window dimensions
        width = win.winfo_width()
        height = win.winfo_height()

        screen_width = win.winfo_screenwidth()
        screen_height = win.winfo_screenheight()

        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)

        # Position the window at the center of the screen
        win.geometry(f"+{x}+{y}")

    app = ClassifierApp(root)

    # Now center the main window
    center_window(root)

    root.mainloop()
