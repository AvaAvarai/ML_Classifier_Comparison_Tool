import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import random
import warnings
import yaml  # Add this to imports at top

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate, KFold
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, recall_score

# Classifiers
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier, Perceptron
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from tkinter import ttk

# For parallel coordinates (matplotlib)
import matplotlib
matplotlib.use("TkAgg")  # Use the TkAgg backend for embedding in Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas.plotting import parallel_coordinates

class ClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning Classifier Comparison Tool")
        
        # Set default window size
        self.root.geometry("1080x720")  # Width x Height

        # Data holders
        self.data = None
        self.class_column = None

        # Optional evaluation data
        self.eval_data = None
        self.eval_class_column = None

        # For storing results
        self.results = []

        # Load parameter configurations from YAML
        try:
            with open('classifier_config.yaml', 'r') as f:
                self.param_config = yaml.safe_load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load classifier configuration: {e}")
            self.root.destroy()
            return

        # Main Notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tabs
        self.file_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.file_frame, text="Data Load and Statistics")

        self.classifiers_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.classifiers_frame, text="Classifier Selection")

        self.params_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.params_tab, text="Classifier Parameters")

        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Results Table")

        # NEW: Plot tab for embedded parallel coordinates
        self.plot_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_tab, text="Results Plot")

        # Add About tab after Plot tab
        self.about_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.about_tab, text="Program About")

        # For storing the FigureCanvas
        self.plot_canvas = None

        # Build each section
        self.build_file_tab()
        self.build_classifiers_tab()
        self.build_params_tab()
        self.build_results_tab()
        self.build_plot_tab()
        self.build_about_tab()

    # ------------------------------------------------------------------------
    # Tab 1: File loading and info
    # ------------------------------------------------------------------------
    def build_file_tab(self):
        # Main file label and load button
        self.file_label = tk.Label(self.file_frame, text="No training data file loaded")
        self.file_label.pack(pady=5)

        self.load_button = tk.Button(
            self.file_frame, text="Load File", command=self.load_file
        )
        self.load_button.pack(pady=5)

        # Evaluation file label and load button
        self.eval_label = tk.Label(self.file_frame, text="No secondary evaluation data file loaded")
        self.eval_label.pack(pady=5)

        self.eval_button = tk.Button(
            self.file_frame, text="Load File (Optional)", command=self.load_eval_file
        )
        self.eval_button.pack(pady=5)

        # Info text (shows main data info only)
        self.info_text = tk.Text(self.file_frame, height=10, state=tk.DISABLED, wrap=tk.WORD)
        self.info_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            self.data = pd.read_csv(file_path)
            self.class_column = self.find_class_column(self.data)
            self.display_dataset_info(self.data, self.class_column)
            self.file_label.config(text=f"Training Data Loaded: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def load_eval_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            self.eval_data = pd.read_csv(file_path)
            self.eval_class_column = self.find_class_column(self.eval_data)
            self.eval_label.config(text=f"Evaluation Data Loaded: {file_path}")
            
            # Update the dataset info to show both datasets
            if hasattr(self, 'data') and self.data is not None:
                self.display_dataset_info(self.data, self.class_column)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load eval file: {e}")

    def find_class_column(self, df):
        for col in df.columns:
            if "class" in col.lower():
                return col
        raise ValueError("No column with 'class' found.")

    def display_dataset_info(self, df, class_col):
        """Display dataset information for both training and evaluation data if available"""
        # Get data type analysis (excluding class column)
        type_analysis, types_count, dataset_type = self.analyze_data_types(df)

        # Get class information
        class_counts = df[class_col].value_counts()
        total_cases = len(df)
        num_classes = len(class_counts)
        balance_ratio = class_counts.min() / class_counts.max() if class_counts.max() != 0 else 0
        majority_class = class_counts.idxmax()

        # Build the info text for training data
        info = (
            f"Training Dataset Statistical Information:\n"
            f"Number of Classes: {num_classes}\n"
            f"Class case counts: {class_counts.to_dict()}\n"
            f"Class balance: {balance_ratio:.2f} (Majority class: {majority_class})\n"
            f"Classes: {', '.join(map(str, df[class_col].unique()))}\n\n"
            f"Dataset Type: {dataset_type.upper()}\n"
            f"Feature Types:\n"
            f"{chr(10).join(f'  {col}: {dtype}' for col, dtype in sorted(type_analysis.items()))}\n\n"
            f"Missing Values: "
            f"{'none' if df.isnull().sum().sum() == 0 else f'contains missing values ({df.isnull().sum().sum()})'}"
        )

        # If evaluation data is loaded, add its information
        if hasattr(self, 'eval_data') and self.eval_data is not None:
            eval_type_analysis, eval_types_count, eval_dataset_type = self.analyze_data_types(self.eval_data)
            eval_class_counts = self.eval_data[self.eval_class_column].value_counts()
            eval_num_classes = len(eval_class_counts)
            eval_balance_ratio = eval_class_counts.min() / eval_class_counts.max() if eval_class_counts.max() != 0 else 0
            eval_majority_class = eval_class_counts.idxmax()

            info += (
                f"\n\n{'-' * 50}\n\n"  # Add separator
                f"Secondary Evaluation Dataset Statistical Information:\n"
                f"Number of Classes: {eval_num_classes}\n"
                f"Class case counts: {eval_class_counts.to_dict()}\n"
                f"Class balance: {eval_balance_ratio:.2f} (Majority class: {eval_majority_class})\n"
                f"Classes: {', '.join(map(str, self.eval_data[self.eval_class_column].unique()))}\n\n"
                f"Dataset Type: {eval_dataset_type.upper()}\n"
                f"Feature Types:\n"
                f"{chr(10).join(f'  {col}: {dtype}' for col, dtype in sorted(eval_type_analysis.items()))}\n\n"
                f"Missing Values: "
                f"{'none' if self.eval_data.isnull().sum().sum() == 0 else f'contains missing values ({self.eval_data.isnull().sum().sum()})'}"
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

        # Create a frame to hold all classifier rows
        classifiers_frame = ttk.Frame(self.classifiers_frame)
        classifiers_frame.pack(fill=tk.X, pady=5)

        # Create a frame for toggle buttons
        toggle_frame = ttk.Frame(classifiers_frame)
        toggle_frame.grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky=tk.W)

        # Add toggle buttons
        self.select_all_button = ttk.Button(
            toggle_frame,
            text="Select All",
            command=self.toggle_all_classifiers
        )
        self.select_all_button.pack(side=tk.LEFT, padx=5)

        self.select_numerical_button = ttk.Button(
            toggle_frame,
            text="Select Numerical",
            command=lambda: self.toggle_by_type("numerical")
        )
        self.select_numerical_button.pack(side=tk.LEFT, padx=5)

        self.select_binary_button = ttk.Button(
            toggle_frame,
            text="Select Binary",
            command=lambda: self.toggle_by_type("binary")
        )
        self.select_binary_button.pack(side=tk.LEFT, padx=5)

        self.select_categorical_button = ttk.Button(
            toggle_frame,
            text="Select Categorical",
            command=lambda: self.toggle_by_type("categorical")
        )
        self.select_categorical_button.pack(side=tk.LEFT, padx=5)

        # Dictionary mapping classifiers to their data types
        self.classifier_types = {
            "Decision Tree": ["numerical", "binary", "categorical"],
            "Random Forest": ["numerical", "binary", "categorical"],
            "Extra Trees": ["numerical", "binary", "categorical"],
            "KNN": ["numerical"],
            "SVM": ["numerical"],
            "LDA": ["numerical"],
            "Logistic Regression": ["numerical"],
            "Ridge": ["numerical"],
            "Naive Bayes": ["numerical"],
            "Bernoulli NB": ["binary"],
            "Multinomial NB": ["categorical"],
            "QDA": ["numerical"],
            "MLP": ["numerical"],
            "SGD": ["numerical"],
            "Gradient Boosting": ["numerical", "binary", "categorical"],
            "AdaBoost": ["numerical", "binary", "categorical"],
            "XGBoost": ["numerical", "binary", "categorical"],
            "CatBoost": ["numerical", "binary", "categorical"],
            "LightGBM": ["numerical", "binary", "categorical"],
            "Passive Aggressive": ["numerical"],
            "Perceptron": ["numerical"]
        }

        # Classifiers with names, acronyms, and data type hints - now in 3 columns
        classifier_rows = [
            # Row 1
            [
                ("Decision Tree (All Types)", "DT"),
                ("Random Forest (All Types)", "RF"),
                ("Extra Trees (All Types)", "ET"),
            ],
            # Row 2
            [
                ("K-Nearest Neighbors (Numerical)", "KNN"),
                ("Support Vector Machine (Numerical)", "SVM"),
                ("Linear Discriminant Analysis (Numerical)", "LDA"),
            ],
            # Row 3
            [
                ("Logistic Regression (Numerical)", "LR"),
                ("Ridge Classifier (Numerical)", "RC"),
                ("Gaussian Naive Bayes (Numerical)", "GNB"),
            ],
            # Row 4
            [
                ("Bernoulli Naive Bayes (Binary)", "BNB"),
                ("Multinomial Naive Bayes (Counts/Text)", "MNB"),
                ("Quadratic Discriminant Analysis (Numerical)", "QDA"),
            ],
            # Row 5
            [
                ("Multi-Layer Perceptron (Numerical)", "MLP"),
                ("Stochastic Gradient Descent (Numerical)", "SGD"),
                ("Gradient Boosting (All Types)", "GB"),
            ],
            # Row 6
            [
                ("AdaBoost (All Types)", "AB"),
                ("XGBoost (All Types)", "XGB"),
                ("CatBoost (Categorical)", "CB"),
            ],
            # Row 7
            [
                ("LightGBM (All Types)", "LGBM"),
                ("Passive Aggressive Classifier (Numerical)", "PA"),
                ("Perceptron (Numerical)", "PCP"),
            ]
        ]

        # Configure grid columns to be equal width
        for i in range(3):
            classifiers_frame.grid_columnconfigure(i, weight=1)

        # Create checkboxes for each classifier - start from row 1 instead of 0
        for row_idx, row_classifiers in enumerate(classifier_rows):
            for col_idx, (clf_display_name, acronym) in enumerate(row_classifiers):
                var = tk.BooleanVar(value=False)
                chk = tk.Checkbutton(
                    classifiers_frame,
                    text=f"{clf_display_name} ({acronym})",
                    variable=var
                )
                chk.grid(row=row_idx + 1, column=col_idx, padx=5, pady=5, sticky=tk.W)  # +1 to row_idx
                
                # Map the short internal name to the variable
                internal_name = {
                    # Row 1
                    "Decision Tree (All Types)": "Decision Tree",
                    "Random Forest (All Types)": "Random Forest",
                    "Extra Trees (All Types)": "Extra Trees",
                    # Row 2
                    "K-Nearest Neighbors (Numerical)": "KNN",
                    "Support Vector Machine (Numerical)": "SVM",
                    "Linear Discriminant Analysis (Numerical)": "LDA",
                    # Row 3
                    "Logistic Regression (Numerical)": "Logistic Regression",
                    "Ridge Classifier (Numerical)": "Ridge",
                    "Gaussian Naive Bayes (Numerical)": "Naive Bayes",
                    # Row 4
                    "Bernoulli Naive Bayes (Binary)": "Bernoulli NB",
                    "Multinomial Naive Bayes (Counts/Text)": "Multinomial NB",
                    "Quadratic Discriminant Analysis (Numerical)": "QDA",
                    # Row 5
                    "Multi-Layer Perceptron (Numerical)": "MLP",
                    "Stochastic Gradient Descent (Numerical)": "SGD",
                    "Gradient Boosting (All Types)": "Gradient Boosting",
                    # Row 6
                    "AdaBoost (All Types)": "AdaBoost",
                    "XGBoost (All Types)": "XGBoost",
                    "CatBoost (Categorical)": "CatBoost",
                    # Row 7
                    "LightGBM (All Types)": "LightGBM",
                    "Passive Aggressive Classifier (Numerical)": "Passive Aggressive",
                    "Perceptron (Numerical)": "Perceptron",
                }[clf_display_name]
                self.selected_classifiers[internal_name] = var

        # Common CV parameters
        common_params_frame = ttk.LabelFrame(
            self.classifiers_frame, text="Cross-Validation and Run Parameters"
        )
        common_params_frame.pack(fill=tk.X, pady=10, padx=5)

        self.cross_val_split = self.add_param_entry(
            parent=common_params_frame,
            label="Cross-Validation Split (>=1, 1=no CV):",
            default="5",
        )
        self.run_count = self.add_param_entry(
            parent=common_params_frame, label="Classifier train and evaluation runs count (>=1):", default="10"
        )
        self.random_seed = self.add_param_entry(
            parent=common_params_frame,
            label="Random number generator seed for CV and classifiers:",
            default=str(random.randint(0, 10**6)),
        )

        # Run button
        self.run_button = tk.Button(
            self.classifiers_frame,
            text="Run Selected Classifiers",
            command=self.run_classifiers,
        )
        self.run_button.pack(pady=5)

        # Add status frame at the bottom
        self.status_frame = ttk.Frame(self.classifiers_frame)
        self.status_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            self.status_frame, 
            mode='determinate',
            length=300
        )
        self.progress_bar.pack(fill=tk.X, pady=2)
        
        # Status label
        self.status_label = ttk.Label(self.status_frame, text="")
        self.status_label.pack(pady=2)

    def toggle_all_classifiers(self):
        """Toggle all classifiers on/off"""
        # Toggle the state
        new_state = not any(var.get() for var in self.selected_classifiers.values())
        
        # Update all checkboxes
        for var in self.selected_classifiers.values():
            var.set(new_state)
        
        # Update button text
        self.select_all_button.config(
            text="Deselect All" if new_state else "Select All"
        )

    def add_param_entry(self, parent, label, default):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)

        lbl = ttk.Label(frame, text=label)
        lbl.pack(side=tk.LEFT, padx=5)

        entry = ttk.Entry(frame, width=8)
        entry.insert(0, default)
        entry.pack(side=tk.LEFT, padx=5)

        return entry

    def toggle_by_type(self, data_type):
        """Toggle all classifiers of a specific data type"""
        # First check if any classifier of this type is unchecked
        any_unchecked = False
        for clf_name, types in self.classifier_types.items():
            if data_type in types and not self.selected_classifiers[clf_name].get():
                any_unchecked = True
                break

        # If any are unchecked, check all. Otherwise, uncheck all
        new_state = any_unchecked
        for clf_name, types in self.classifier_types.items():
            if data_type in types:
                self.selected_classifiers[clf_name].set(new_state)
        
        # Update button text based on data type
        if data_type == "numerical":
            self.select_numerical_button.config(
                text="Deselect Numerical" if new_state else "Select Numerical"
            )
        elif data_type == "binary":
            self.select_binary_button.config(
                text="Deselect Binary" if new_state else "Select Binary"
            )
        elif data_type == "categorical":
            self.select_categorical_button.config(
                text="Deselect Categorical" if new_state else "Select Categorical"
            )

    # ------------------------------------------------------------------------
    # Tab 3: Hyperparameter controls
    # ------------------------------------------------------------------------
    def build_params_tab(self):
        canvas = tk.Canvas(self.params_tab)
        scrollbar = ttk.Scrollbar(self.params_tab, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Add mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', _bind_mousewheel)
        canvas.bind('<Leave>', _unbind_mousewheel)

        # Create left and right columns
        left_column = ttk.Frame(scroll_frame)
        right_column = ttk.Frame(scroll_frame)
        left_column.grid(row=0, column=0, padx=5, pady=5, sticky="n")
        right_column.grid(row=0, column=1, padx=5, pady=5, sticky="n")

        # Configure grid to make columns equal width
        scroll_frame.grid_columnconfigure(0, weight=1)
        scroll_frame.grid_columnconfigure(1, weight=1)

        self.hyperparam_entries = {}
        # Split classifiers into two groups
        classifier_names = list(self.param_config.keys())
        mid_point = (len(classifier_names) + 1) // 2

        # Left column classifiers
        for clf_name in classifier_names[:mid_point]:
            group = ttk.LabelFrame(left_column, text=clf_name)
            group.pack(fill=tk.X, padx=5, pady=5)
            
            self.hyperparam_entries[clf_name] = {}
            for param_name, info in self.param_config[clf_name].items():
                self._create_param_widget(group, clf_name, param_name, info)

        # Right column classifiers
        for clf_name in classifier_names[mid_point:]:
            group = ttk.LabelFrame(right_column, text=clf_name)
            group.pack(fill=tk.X, padx=5, pady=5)
            
            self.hyperparam_entries[clf_name] = {}
            for param_name, info in self.param_config[clf_name].items():
                self._create_param_widget(group, clf_name, param_name, info)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_param_widget(self, group, clf_name, param_name, info):
        """Helper method to create parameter widgets"""
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

        elif param_type in ["numeric", "text"]:
            entry = ttk.Entry(row_frame, width=8)
            entry.insert(0, str(default_value))
            entry.pack(side=tk.LEFT, padx=5)
            self.hyperparam_entries[clf_name][param_name] = entry

        help_label = ttk.Label(row_frame, text=f"({param_help})", foreground="gray")
        help_label.pack(side=tk.LEFT, padx=5)

    # ------------------------------------------------------------------------
    # Tab 4: Results
    # ------------------------------------------------------------------------
    def build_results_tab(self):
        """
        We'll keep columns for best/worst/avg/std for accuracy, F1, recall.
        This remains the same structure for either main-data CV or eval-data CV.
        """
        # Add sorting state variables
        self.sort_column = None
        self.sort_reverse = False

        self.results_table = ttk.Treeview(
            self.results_tab,
            columns=(
                "Classifier",
                # ACC
                "ACC Best", "ACC Worst", "ACC Avg", "ACC Std",
                # F1
                "F1 Best", "F1 Worst", "F1 Avg", "F1 Std",
                # Recall
                "REC Best", "REC Worst", "REC Avg", "REC Std"
            ),
            show="headings",
            height=10
        )

        # Configure column headings with click binding
        for col in self.results_table["columns"]:
            self.results_table.heading(
                col, 
                text=col,
                command=lambda c=col: self.sort_results_by(c)
            )
            self.results_table.column(col, width=80, anchor=tk.CENTER)

        self.results_table.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)

        # Export button
        self.export_button = tk.Button(
            self.results_tab,
            text="Export to CSV",
            command=self.export_results
        )
        self.export_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Visualize button
        self.visualize_button = tk.Button(
            self.results_tab,
            text="Visualize",
            command=self.visualize_results
        )
        self.visualize_button.pack(side=tk.LEFT, padx=5, pady=5)

    def sort_results_by(self, column):
        """Sort results table by the selected column"""
        # If clicking the same column, reverse the sort order
        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = column
            self.sort_reverse = False

        # Get all items
        items = [(self.results_table.set(item, column), item) for item in self.results_table.get_children("")]
        
        # Sort items
        if column != "Classifier":
            # For numeric columns, extract the actual value before the parentheses
            items = [(float(val.split()[0]), item) for val, item in items]
            items.sort(reverse=self.sort_reverse)
            items = [(str(val), item) for val, item in items]
        else:
            # For Classifier column, sort alphabetically
            items.sort(reverse=self.sort_reverse)

        # Rearrange items in sorted order
        for index, (_, item) in enumerate(items):
            self.results_table.move(item, "", index)

        # Add visual feedback - update column header
        for col in self.results_table["columns"]:
            if col == column:
                # Add arrow to indicate sort direction
                arrow = "↓" if self.sort_reverse else "↑"
                self.results_table.heading(col, text=f"{col} {arrow}")
            else:
                # Remove arrow from other columns
                self.results_table.heading(col, text=col)

    # ------------------------------------------------------------------------
    # Tab 5: Plot (for embedded parallel coordinates)
    # ------------------------------------------------------------------------
    def build_plot_tab(self):
        """
        Build the Plot tab with toggle buttons and export button
        """
        # Frame for controls
        controls_frame = ttk.Frame(self.plot_tab)
        controls_frame.pack(fill=tk.X, pady=5)

        # Normalization toggle button
        self.normalize_var = tk.BooleanVar(value=True)
        self.normalize_button = ttk.Button(
            controls_frame,
            text="Toggle Normalization (On)",
            command=self.toggle_normalization
        )
        self.normalize_button.pack(side=tk.LEFT, padx=5)

        # Show axes toggle button - change default to False
        self.show_axes_var = tk.BooleanVar(value=False)  # Default to no axes
        self.show_axes_button = ttk.Button(
            controls_frame,
            text="Toggle Axes (Off)",  # Update initial text
            command=self.toggle_axes
        )
        self.show_axes_button.pack(side=tk.LEFT, padx=5)

        # Export plot button
        self.export_plot_button = ttk.Button(
            controls_frame,
            text="Export Plot",
            command=self.export_plot
        )
        self.export_plot_button.pack(side=tk.LEFT, padx=5)

        # Placeholder label for plot area
        self.plot_placeholder = tk.Label(self.plot_tab, text="Parallel Coordinates will be shown here.")
        self.plot_placeholder.pack(pady=10)

    def toggle_normalization(self):
        """Toggle normalization and update the visualization"""
        self.normalize_var.set(not self.normalize_var.get())
        # Update button text
        self.normalize_button.config(
            text=f"Toggle Normalization ({'On' if self.normalize_var.get() else 'Off'})"
        )
        # Redraw the visualization if we have results
        if hasattr(self, 'train_results') and self.train_results:
            self.visualize_results()

    def toggle_axes(self):
        """Toggle axes visibility and update the visualization"""
        self.show_axes_var.set(not self.show_axes_var.get())
        # Update button text
        self.show_axes_button.config(
            text=f"Toggle Axes ({'On' if self.show_axes_var.get() else 'Off'})"
        )
        # Redraw the visualization if we have results
        if hasattr(self, 'train_results') and self.train_results:
            self.visualize_results()

    def visualize_results(self):
        """Build parallel coordinates plots for training and evaluation results"""
        if not hasattr(self, 'train_results') or not self.train_results:
            messagebox.showerror("Error", "No results to visualize.")
            return

        # Convert results to DataFrames
        columns = [
            "Classifier",
            "ACC Best", "ACC Worst", "ACC Avg", "ACC Std",
            "F1 Best", "F1 Worst", "F1 Avg", "F1 Std",
            "REC Best", "REC Worst", "REC Avg", "REC Std"
        ]
        
        train_df = pd.DataFrame(self.train_results, columns=columns)
        train_df['Dataset'] = 'Training'
        
        if hasattr(self, 'eval_results') and self.eval_results:
            eval_df = pd.DataFrame(self.eval_results, columns=columns)
            eval_df['Dataset'] = 'Evaluation'
            # Combine the dataframes
            df = pd.concat([train_df, eval_df], ignore_index=True)
        else:
            df = train_df

        # Check normalization toggle
        numerical_cols = [
            "ACC Best", "ACC Worst", "ACC Avg", "ACC Std",
            "F1 Best", "F1 Worst", "F1 Avg", "F1 Std",
            "REC Best", "REC Worst", "REC Avg", "REC Std"
        ]
        
        if self.normalize_var.get():
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        # Clear the previous plot if it exists
        if self.plot_canvas:
            self.plot_canvas.get_tk_widget().destroy()
            plt.close('all')

        # Create the figure with a smaller width
        fig, ax = plt.subplots(figsize=(8, 6))

        # Define specific colors and line styles
        colors = [
            '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', 
            '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', 
            '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', 
            '#000075', '#808080', '#ffffff', '#000000'
        ]
        line_styles = ['-', '--', ':', '-.']  # Used after colors are exhausted
        
        # Create unique combinations of colors and line styles
        style_cycler = []
        num_classifiers = len(df)
        
        for i in range(num_classifiers):
            if i < len(colors):
                # Use unique color with solid line
                style_cycler.append({
                    'color': colors[i],
                    'linestyle': '-'
                })
            else:
                # After colors are exhausted, cycle through colors with different line styles
                color_idx = i % len(colors)
                style_idx = (i // len(colors)) % len(line_styles)
                style_cycler.append({
                    'color': colors[color_idx],
                    'linestyle': line_styles[style_idx]
                })

        # Plot each classifier's line with a unique style
        for idx, (classifier, dataset) in enumerate(df.groupby(['Classifier', 'Dataset'])):
            classifier_name, dataset_type = classifier  # Unpack the tuple
            classifier_data = df[(df['Classifier'] == classifier_name) & (df['Dataset'] == dataset_type)]
            
            # Create label with classifier name and dataset type
            label = f"{classifier_name} ({dataset_type})"
            
            # Plot the parallel coordinates for this classifier
            for i in range(len(numerical_cols)-1):
                ax.plot(
                    [i, i+1],
                    [classifier_data[numerical_cols[i]], classifier_data[numerical_cols[i+1]]],  # Fixed brackets
                    label=label if i == 0 else "_nolegend_",
                    **style_cycler[idx]
                )

        # Always keep the border (spines) and labels
        for spine in ax.spines.values():
            spine.set_visible(True)
        
        # Always show x and y labels
        ax.set_xticks(range(len(numerical_cols)))
        ax.set_xticklabels(numerical_cols, rotation=45, ha='right')
        ax.yaxis.set_visible(True)

        # Toggle parallel axes visibility
        if self.show_axes_var.get():
            # Draw parallel coordinate axes
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Draw vertical lines for each dimension
            for i in range(len(numerical_cols)):
                ax.axvline(x=i, color='gray', linestyle='-', alpha=0.5)
        else:
            # Hide just the grid
            ax.grid(False)

        # Adjust the legend position
        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                          ncol=3, frameon=False, title="Classifiers")

        # Add a title and labels
        ax.set_title("Parallel Coordinates: Classifier Evaluation Metrics")
        ax.set_ylabel("Normalized Evaluation Metric Value" if self.normalize_var.get() else "Evaluation Metric Value")
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Remove the placeholder label if it exists
        if hasattr(self, 'plot_placeholder') and self.plot_placeholder:
            self.plot_placeholder.destroy()
            self.plot_placeholder = None

        # Embed the figure in the plot tab
        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_tab)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Switch to the Plot tab
        self.notebook.select(self.plot_tab)

    def export_plot(self):
        """Save the current plot to a PNG file"""
        if not hasattr(self, 'plot_canvas') or not self.plot_canvas:
            messagebox.showerror("Error", "No plot to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")]
        )
        if not file_path:
            return

        try:
            # Get the figure from the canvas
            fig = self.plot_canvas.figure
            
            # Save with tight layout to prevent cutoff
            fig.savefig(
                file_path,
                bbox_inches='tight',
                dpi=300
            )
            messagebox.showinfo("Success", "Plot exported successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export plot: {str(e)}")

    # ------------------------------------------------------------------------
    # Running classifiers
    # ------------------------------------------------------------------------
    def run_classifiers(self):
        if self.data is None:
            messagebox.showerror("Error", "No main dataset loaded.")
            return

        try:
            num_splits = int(self.cross_val_split.get())
            if num_splits < 1:
                raise ValueError("Cross-validation splits must be at least 1")

            run_count = int(self.run_count.get())
            if run_count < 1:
                raise ValueError("Run count must be at least 1")

            random_seed = int(self.random_seed.get())

            # Main data
            X_main = self.data.drop(columns=self.class_column)
            y_main = self.data[self.class_column]

            # Label-encode main data if needed
            if y_main.dtype == object or y_main.dtype.kind in {'U', 'S'}:
                self.label_encoder = LabelEncoder()
                y_main = self.label_encoder.fit_transform(y_main)

            # Check if we have optional eval data
            use_eval = (self.eval_data is not None)
            if use_eval:
                X_eval = self.eval_data.drop(columns=self.eval_class_column)
                y_eval = self.eval_data[self.eval_class_column]
                # Encode eval data with same encoder
                if y_eval.dtype == object or y_eval.dtype.kind in {'U', 'S'}:
                    y_eval = self.label_encoder.transform(y_eval)

            # Create separate lists for training and evaluation results
            self.train_results = []
            self.eval_results = []
            convergence_issues = set()

            # Count total operations for progress bar - double if using eval data
            selected_clf_count = sum(1 for var in self.selected_classifiers.values() if var.get())
            total_operations = selected_clf_count * run_count * (2 if use_eval else 1)
            current_operation = 0
            
            # Reset and show progress bar
            self.progress_bar['value'] = 0
            self.progress_bar['maximum'] = total_operations
            
            for clf_name, var in self.selected_classifiers.items():
                if not var.get():
                    continue

                # Update status to show classifier name and total runs
                self.status_label['text'] = f"Training - Running {clf_name} (0/{run_count} runs)..."
                self.root.update()

                hyperparams = self.parse_hyperparams(clf_name)
                classifier = self.build_classifier(clf_name, hyperparams, random_seed)

                # Training results collection
                train_accuracy_scores = []
                train_f1_scores = []
                train_recall_scores = []

                # Evaluation results collection (if using eval data)
                eval_accuracy_scores = []
                eval_f1_scores = []
                eval_recall_scores = []

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", ConvergenceWarning)

                    for run_i in range(run_count):
                        current_operation += 1
                        self.progress_bar['value'] = current_operation
                        self.status_label['text'] = f"Training - Running {clf_name} ({run_i + 1}/{run_count} runs)..."
                        self.root.update()

                        seed_for_run = random_seed + run_i
                        np.random.seed(seed_for_run)
                        random.seed(seed_for_run)

                        if num_splits == 1:
                            # Train on full training data
                            classifier.random_state = seed_for_run
                            classifier.fit(X_main, y_main)

                            # Get training scores
                            train_preds = classifier.predict(X_main)
                            train_accuracy_scores.append(accuracy_score(y_main, train_preds))
                            train_f1_scores.append(f1_score(y_main, train_preds, average='macro'))
                            train_recall_scores.append(recall_score(y_main, train_preds, average='macro'))

                            # If evaluation data exists, get evaluation scores
                            if use_eval:
                                eval_preds = classifier.predict(X_eval)
                                eval_accuracy_scores.append(accuracy_score(y_eval, eval_preds))
                                eval_f1_scores.append(f1_score(y_eval, eval_preds, average='macro'))
                                eval_recall_scores.append(recall_score(y_eval, eval_preds, average='macro'))

                        else:
                            # Cross-validation
                            cv = KFold(n_splits=num_splits, shuffle=True, random_state=seed_for_run)
                            
                            # Training data CV
                            scores = cross_validate(
                                classifier,
                                X_main,
                                y_main,
                                cv=cv,
                                scoring={'acc': 'accuracy', 'f1': 'f1_macro', 'rec': 'recall_macro'},
                                return_train_score=False
                            )
                            train_accuracy_scores.extend(scores['test_acc'])
                            train_f1_scores.extend(scores['test_f1'])
                            train_recall_scores.extend(scores['test_rec'])

                            # If evaluation data exists, do CV on it too
                            if use_eval:
                                eval_cv = KFold(n_splits=num_splits, shuffle=True, random_state=seed_for_run)
                                for train_idx, test_idx in eval_cv.split(X_eval):
                                    # Train on full training data
                                    classifier.fit(X_main, y_main)
                                    # Evaluate on eval data fold
                                    X_eval_test = X_eval.iloc[test_idx]
                                    y_eval_test = y_eval[test_idx]
                                    eval_preds = classifier.predict(X_eval_test)
                                    eval_accuracy_scores.append(accuracy_score(y_eval_test, eval_preds))
                                    eval_f1_scores.append(f1_score(y_eval_test, eval_preds, average='macro'))
                                    eval_recall_scores.append(recall_score(y_eval_test, eval_preds, average='macro'))

                # Add training results
                self.train_results.append((
                    clf_name,
                    max(train_accuracy_scores), min(train_accuracy_scores), 
                    np.mean(train_accuracy_scores), np.std(train_accuracy_scores),
                    max(train_f1_scores), min(train_f1_scores), 
                    np.mean(train_f1_scores), np.std(train_f1_scores),
                    max(train_recall_scores), min(train_recall_scores), 
                    np.mean(train_recall_scores), np.std(train_recall_scores)
                ))

                # Add evaluation results if they exist
                if use_eval:
                    self.eval_results.append((
                        clf_name,
                        max(eval_accuracy_scores), min(eval_accuracy_scores), 
                        np.mean(eval_accuracy_scores), np.std(eval_accuracy_scores),
                        max(eval_f1_scores), min(eval_f1_scores), 
                        np.mean(eval_f1_scores), np.std(eval_f1_scores),
                        max(eval_recall_scores), min(eval_recall_scores), 
                        np.mean(eval_recall_scores), np.std(eval_recall_scores)
                    ))

            # Update results display
            self.update_results_table()

            # Clear status when done
            self.status_label['text'] = "Completed!"
            self.progress_bar['value'] = total_operations
            self.root.update()

            if convergence_issues:
                messagebox.showinfo(
                    "Convergence Notice",
                    f"The following classifiers reached max_iter before converging:\n"
                    f"{', '.join(sorted(convergence_issues))}\n\n"
                    f"Consider increasing max_iter or adjusting other hyperparameters for these classifiers."
                )

            self.notebook.select(self.results_tab)

        except ValueError as e:
            messagebox.showerror("Error", str(e))
            # Debug print
            # print(f"ValueError: {e}")  # Debug print
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            # Debug print
            # print(f"Unexpected Error: {e}")  # Debug print

    # ------------------------------------------------------------------------
    # Parsing hyperparams
    # ------------------------------------------------------------------------
    def parse_hyperparams(self, clf_name):
        param_values = {}
        config = self.param_config[clf_name]
        for param_name, widget in self.hyperparam_entries[clf_name].items():
            param_info = config[param_name]
            param_type = param_info["type"]
            raw_val = widget.get().strip()

            if param_type == "combo":
                param_values[param_name] = raw_val
            elif param_type in ["numeric", "text"]:
                if raw_val.lower() == "none":
                    param_values[param_name] = None
                else:
                    # Try int
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
                    # If parentheses => evaluate
                    if raw_val.startswith("(") or raw_val.startswith("["):
                        try:
                            param_values[param_name] = eval(raw_val)
                            continue
                        except:
                            pass
                    # Otherwise store string
                    param_values[param_name] = raw_val
        return param_values

    def build_classifier(self, clf_name, hyperparams, random_seed):
        """Instantiate the classifier with the parsed hyperparameters."""
        # Debug print
        # print(f"Building classifier: {clf_name} with parameters: {hyperparams}")
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
        elif clf_name == "QDA":
            return QuadraticDiscriminantAnalysis(
                **{k: v for k, v in hyperparams.items() if k in ["reg_param"]}
            )
        elif clf_name == "Bernoulli NB":
            return BernoulliNB(
                **{k: v for k, v in hyperparams.items() if k in ["alpha", "binarize"]}
            )
        elif clf_name == "Multinomial NB":
            # Convert string 'True'/'False' to boolean if needed
            if 'fit_prior' in hyperparams:
                hyperparams['fit_prior'] = hyperparams['fit_prior'] in [True, 'True']
            return MultinomialNB(
                **{k: v for k, v in hyperparams.items() if k in ["alpha", "fit_prior"]}
            )
        elif clf_name == "CatBoost":
            return CatBoostClassifier(
                random_state=random_seed,
                **{k: v for k, v in hyperparams.items() if k in ["iterations", "learning_rate", "depth"]}
            )
        elif clf_name == "LightGBM":
            return LGBMClassifier(
                random_state=random_seed,
                **{k: v for k, v in hyperparams.items() if k in ["n_estimators", "learning_rate", "max_depth"]}
            )
        elif clf_name == "Passive Aggressive":
            return PassiveAggressiveClassifier(
                random_state=random_seed,
                **{k: v for k, v in hyperparams.items() if k in ["C", "max_iter"]}
            )
        elif clf_name == "Perceptron":
            return Perceptron(
                random_state=random_seed,
                **{k: v for k, v in hyperparams.items() if k in ["alpha", "max_iter"]}
            )
        else:
            raise ValueError(f"Unknown classifier: {clf_name}")

    # ------------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------------
    def update_results_table(self):
        # Clear old rows
        for row in self.results_table.get_children():
            self.results_table.delete(row)

        # Add training results with evaluation differences if they exist
        if hasattr(self, 'train_results') and self.train_results:
            for i, train_res in enumerate(self.train_results):
                (
                    clf_name,
                    train_best_acc, train_worst_acc, train_avg_acc, train_std_acc,
                    train_best_f1, train_worst_f1, train_avg_f1, train_std_f1,
                    train_best_rec, train_worst_rec, train_avg_rec, train_std_rec
                ) = train_res

                # If evaluation results exist, calculate differences
                if hasattr(self, 'eval_results') and self.eval_results:
                    eval_res = self.eval_results[i]
                    (
                        _, # clf_name same as training
                        eval_best_acc, eval_worst_acc, eval_avg_acc, eval_std_acc,
                        eval_best_f1, eval_worst_f1, eval_avg_f1, eval_std_f1,
                        eval_best_rec, eval_worst_rec, eval_avg_rec, eval_std_rec
                    ) = eval_res

                    # Format values with differences
                    row_values = (
                        clf_name,
                        f"{train_best_acc:.4f} ({self._format_diff(eval_best_acc - train_best_acc)})",
                        f"{train_worst_acc:.4f} ({self._format_diff(eval_worst_acc - train_worst_acc)})",
                        f"{train_avg_acc:.4f} ({self._format_diff(eval_avg_acc - train_avg_acc)})",
                        f"{train_std_acc:.4f} ({self._format_diff(eval_std_acc - train_std_acc)})",
                        f"{train_best_f1:.4f} ({self._format_diff(eval_best_f1 - train_best_f1)})",
                        f"{train_worst_f1:.4f} ({self._format_diff(eval_worst_f1 - train_worst_f1)})",
                        f"{train_avg_f1:.4f} ({self._format_diff(eval_avg_f1 - train_avg_f1)})",
                        f"{train_std_f1:.4f} ({self._format_diff(eval_std_f1 - train_std_f1)})",
                        f"{train_best_rec:.4f} ({self._format_diff(eval_best_rec - train_best_rec)})",
                        f"{train_worst_rec:.4f} ({self._format_diff(eval_worst_rec - train_worst_rec)})",
                        f"{train_avg_rec:.4f} ({self._format_diff(eval_avg_rec - train_avg_rec)})",
                        f"{train_std_rec:.4f} ({self._format_diff(eval_std_rec - train_std_rec)})"
                    )
                else:
                    # Just show training values without differences
                    row_values = (
                        clf_name,
                        f"{train_best_acc:.4f}",
                        f"{train_worst_acc:.4f}",
                        f"{train_avg_acc:.4f}",
                        f"{train_std_acc:.4f}",
                        f"{train_best_f1:.4f}",
                        f"{train_worst_f1:.4f}",
                        f"{train_avg_f1:.4f}",
                        f"{train_std_f1:.4f}",
                        f"{train_best_rec:.4f}",
                        f"{train_worst_rec:.4f}",
                        f"{train_avg_rec:.4f}",
                        f"{train_std_rec:.4f}"
                    )
                
                self.results_table.insert("", "end", values=row_values)

    def _format_diff(self, diff):
        """Format difference with + or - sign"""
        if diff > 0:
            return f"+{diff:.4f}"
        else:
            return f"{diff:.4f}"

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

        df = pd.DataFrame(self.results, columns=[
            "Classifier",
            "ACC Best", "ACC Worst", "ACC Avg", "ACC Std",
            "F1 Best", "F1 Worst", "F1 Avg", "F1 Std",
            "REC Best", "REC Worst", "REC Avg", "REC Std"
        ])
        df.to_csv(file_path, index=False)
        messagebox.showinfo("Success", "Results exported successfully.")

    def cleanup(self):
        # Properly close and clean up matplotlib figures
        if self.plot_canvas:
            plt.close(self.plot_canvas.figure)
        self.root.destroy()

    def analyze_data_types(self, data):
        """Analyze data types of columns and overall dataset"""
        type_analysis = {}
        
        for column in data.columns:
            # Skip the target/class column
            if column == self.class_column:
                continue
            
            # Get unique values
            unique_vals = data[column].nunique()
            
            # Check if boolean/binary
            if data[column].dtype in ['bool'] or (unique_vals == 2):
                type_analysis[column] = "binary"
            
            # Check if categorical
            elif data[column].dtype in ['object', 'category']:
                type_analysis[column] = "categorical"
            
            # Must be numerical
            else:
                type_analysis[column] = "numerical"
        
        # Count feature types (excluding class column)
        types_count = {
            "numerical": sum(1 for t in type_analysis.values() if t == "numerical"),
            "categorical": sum(1 for t in type_analysis.values() if t == "categorical"),
            "binary": sum(1 for t in type_analysis.values() if t == "binary")
        }
        
        # If we have more than one type, it's mixed
        unique_types = set(type_analysis.values())
        dataset_type = "mixed" if len(unique_types) > 1 else list(unique_types)[0]
        
        return type_analysis, types_count, dataset_type

    def build_about_tab(self):
        """Build the About tab with application information"""
        # Create a Text widget for the about information
        about_text = tk.Text(
            self.about_tab,
            wrap=tk.WORD,
            height=20,
            width=60,
            padx=10,
            pady=10
        )
        about_text.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        try:
            # Load about text from file
            with open('about.txt', 'r') as f:
                info = f.read()
        except Exception as e:
            info = "Error loading about information."
            messagebox.showerror("Error", f"Failed to load about information: {e}")

        # Insert the text and disable editing
        about_text.insert('1.0', info)
        about_text.config(state=tk.DISABLED)

        # Add a scrollbar
        scrollbar = ttk.Scrollbar(self.about_tab, orient=tk.VERTICAL, command=about_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        about_text.config(yscrollcommand=scrollbar.set)


if __name__ == "__main__":
    root = tk.Tk()

    # Center the window
    def center_window(win):
        win.update_idletasks()
        width = win.winfo_width()
        height = win.winfo_height()
        screen_width = win.winfo_screenwidth()
        screen_height = win.winfo_screenheight()

        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        win.geometry(f"+{x}+{y}")

    app = ClassifierApp(root)
    center_window(root)

    # Bind cleanup to the close event
    root.protocol("WM_DELETE_WINDOW", app.cleanup)

    root.mainloop()
