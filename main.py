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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

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
        self.notebook.add(self.file_frame, text="Data")

        self.classifiers_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.classifiers_frame, text="Classifiers")

        self.params_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.params_tab, text="Parameters")

        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Results")

        # NEW: Plot tab for embedded parallel coordinates
        self.plot_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_tab, text="Plot")

        # For storing the FigureCanvas
        self.plot_canvas = None

        # Build each section
        self.build_file_tab()
        self.build_classifiers_tab()
        self.build_params_tab()
        self.build_results_tab()
        self.build_plot_tab()  # minimal UI for the "Plot" tab

    # ------------------------------------------------------------------------
    # Tab 1: File loading and info
    # ------------------------------------------------------------------------
    def build_file_tab(self):
        # Main file label and load button
        self.file_label = tk.Label(self.file_frame, text="No file loaded")
        self.file_label.pack(pady=5)

        self.load_button = tk.Button(
            self.file_frame, text="Load File", command=self.load_file
        )
        self.load_button.pack(pady=5)

        # Evaluation file label and load button
        self.eval_label = tk.Label(self.file_frame, text="No evaluate file loaded")
        self.eval_label.pack(pady=5)

        self.eval_button = tk.Button(
            self.file_frame, text="Load Evaluate File", command=self.load_eval_file
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
            self.file_label.config(text=f"Loaded: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def load_eval_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            self.eval_data = pd.read_csv(file_path)
            self.eval_class_column = self.find_class_column(self.eval_data)
            self.eval_label.config(text=f"Loaded eval data: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load eval file: {e}")

    def find_class_column(self, df):
        for col in df.columns:
            if "class" in col.lower():
                return col
        raise ValueError("No column with 'class' found.")

    def display_dataset_info(self, df, class_col):
        class_counts = df[class_col].value_counts()
        total_cases = len(df)
        num_classes = len(class_counts)
        balance_ratio = class_counts.min() / class_counts.max() if class_counts.max() != 0 else 0

        info = (
            f"Classes: {list(class_counts.index)}\n"
            f"Class Counts: {class_counts.to_dict()}\n"
            f"Total Cases: {total_cases}\n"
            f"Number of Classes: {num_classes}\n"
            f"Class Balance: {'Balanced' if abs(balance_ratio - 1.0) < 1e-9 else 'Not Balanced'} "
            f"(Ratio: {balance_ratio:.2f})\n"
            f"Number of Features: {len(df.columns) - 1}\n"
            f"Features: {[col for col in df.columns if col != class_col]}\n"
            f"Missing Values: "
            f"{'none' if df.isnull().sum().sum() == 0 else f'contains missing values ({df.isnull().sum().sum()})'}"
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

        # Add select all toggle button at the top
        self.select_all_var = tk.BooleanVar(value=False)
        self.select_all_button = ttk.Button(
            classifiers_frame,
            text="Select All",
            command=self.toggle_all_classifiers
        )
        self.select_all_button.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        # Classifiers with names and acronyms - now in 4 rows
        classifier_rows = [
            # Row 1
            [
                ("Decision Tree", "DT"),
                ("Random Forest", "RF"),
                ("Extra Trees", "ET"),
                ("KNN", "KNN"),
            ],
            # Row 2
            [
                ("Support Vector Machine", "SVM"),
                ("Linear Discriminant Analysis", "LDA"),
                ("Logistic Regression", "LR"),
                ("Ridge Classifier", "RC"),
            ],
            # Row 3
            [
                ("Gaussian Naive Bayes", "GNB"),
                ("Multi-Layer Perceptron", "MLP"),
                ("SGD", "SGD"),
                ("Gradient Boosting Classifier", "GB"),
            ],
            # Row 4
            [
                ("AdaBoost Classifier", "AB"),
                ("XGBoost Classifier", "XGB"),
            ]
        ]

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
                    "Decision Tree": "Decision Tree",
                    "Random Forest": "Random Forest",
                    "Extra Trees": "Extra Trees",
                    "KNN": "KNN",
                    # Row 2
                    "Support Vector Machine": "SVM",
                    "Linear Discriminant Analysis": "LDA",
                    "Logistic Regression": "Logistic Regression",
                    "Ridge Classifier": "Ridge",
                    # Row 3
                    "Gaussian Naive Bayes": "Naive Bayes",
                    "Multi-Layer Perceptron": "MLP",
                    "SGD": "SGD",
                    "Gradient Boosting Classifier": "Gradient Boosting",
                    # Row 4
                    "AdaBoost Classifier": "AdaBoost",
                    "XGBoost Classifier": "XGBoost",
                }[clf_display_name]
                self.selected_classifiers[internal_name] = var

        # Common CV parameters
        common_params_frame = ttk.LabelFrame(
            self.classifiers_frame, text="Common CV Parameters"
        )
        common_params_frame.pack(fill=tk.X, pady=10, padx=5)

        self.cross_val_split = self.add_param_entry(
            parent=common_params_frame,
            label="Cross-Validation Split (>=1, 1=no CV):",
            default="5",
        )
        self.run_count = self.add_param_entry(
            parent=common_params_frame, label="Run Count (>=1):", default="10"
        )
        self.random_seed = self.add_param_entry(
            parent=common_params_frame,
            label="Random Seed:",
            default=str(random.randint(0, 10**6)),
        )

        # Run button
        self.run_button = tk.Button(
            self.classifiers_frame,
            text="Run Selected Classifiers",
            command=self.run_classifiers,
        )
        self.run_button.pack(pady=5)

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

        for col in self.results_table["columns"]:
            self.results_table.heading(col, text=col)
            self.results_table.column(col, width=80, anchor=tk.CENTER)

        self.results_table.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)

        # Export button
        self.export_button = tk.Button(
            self.results_tab,
            text="Export to CSV",
            command=self.export_results
        )
        self.export_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Visualize button => calls self.visualize_results
        self.visualize_button = tk.Button(
            self.results_tab,
            text="Visualize",
            command=self.visualize_results
        )
        self.visualize_button.pack(side=tk.LEFT, padx=5, pady=5)

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
        if hasattr(self, 'results') and self.results:
            self.visualize_results()

    def toggle_axes(self):
        """Toggle axes visibility and update the visualization"""
        self.show_axes_var.set(not self.show_axes_var.get())
        # Update button text
        self.show_axes_button.config(
            text=f"Toggle Axes ({'On' if self.show_axes_var.get() else 'Off'})"
        )
        # Redraw the visualization if we have results
        if hasattr(self, 'results') and self.results:
            self.visualize_results()

    def visualize_results(self):
        """
        Build a parallel coordinates plot from self.results,
        then embed it in the "Plot" tab via FigureCanvasTkAgg.
        """
        if not self.results:
            messagebox.showerror("Error", "No results to visualize.")
            return

        # Convert self.results => DataFrame
        columns = [
            "Classifier",
            "ACC Best", "ACC Worst", "ACC Avg", "ACC Std",
            "F1 Best", "F1 Worst", "F1 Avg", "F1 Std",
            "REC Best", "REC Worst", "REC Avg", "REC Std"
        ]
        df = pd.DataFrame(self.results, columns=columns)

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

        # Define line styles and colors
        line_styles = ['-', '--', ':', '-.']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Create unique combinations of colors and line styles
        style_cycler = []
        num_classifiers = len(df)
        for i in range(num_classifiers):
            color_idx = i % len(colors)
            style_idx = i // len(colors)
            style_cycler.append({
                'color': colors[color_idx],
                'linestyle': line_styles[style_idx % len(line_styles)]
            })

        # Plot each classifier's line with a unique style
        for idx, classifier in enumerate(df['Classifier'].unique()):
            classifier_data = df[df['Classifier'] == classifier]
            
            # Plot the parallel coordinates for this classifier
            for i in range(len(numerical_cols)-1):
                ax.plot(
                    [i, i+1],
                    [classifier_data[numerical_cols[i]], classifier_data[numerical_cols[i+1]]],
                    label=classifier if i == 0 else "_nolegend_",
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
        ax.set_title("Parallel Coordinates: Classifier Metrics")
        ax.set_ylabel("Normalized Metric Value" if self.normalize_var.get() else "Metric Value")
        
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

            self.results = []
            convergence_issues = set()  # To store names of classifiers with convergence issues

            for clf_name, var in self.selected_classifiers.items():
                if not var.get():
                    continue

                hyperparams = self.parse_hyperparams(clf_name)
                classifier = self.build_classifier(clf_name, hyperparams, random_seed)

                accuracy_scores = []
                f1_scores = []
                recall_scores = []

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", ConvergenceWarning)

                    for run_i in range(run_count):
                        seed_for_run = random_seed + run_i
                        np.random.seed(seed_for_run)
                        random.seed(seed_for_run)

                        if num_splits == 1:
                            # NO CROSS-VALIDATION
                            classifier.random_state = seed_for_run
                            classifier.fit(X_main, y_main)

                            if use_eval:
                                preds = classifier.predict(X_eval)
                                true_labels = y_eval
                            else:
                                preds = classifier.predict(X_main)
                                true_labels = y_main

                            acc_ = accuracy_score(true_labels, preds)
                            f1_ = f1_score(true_labels, preds, average='macro')
                            rec_ = recall_score(true_labels, preds, average='macro')

                            accuracy_scores.append(acc_)
                            f1_scores.append(f1_)
                            recall_scores.append(rec_)

                        else:
                            # CROSS-VALIDATION
                            cv = KFold(n_splits=num_splits, shuffle=True, random_state=seed_for_run)

                            if not use_eval:
                                # CV on main data
                                scores = cross_validate(
                                    classifier,
                                    X_main,
                                    y_main,
                                    cv=cv,
                                    scoring={'acc': 'accuracy', 'f1': 'f1_macro', 'rec': 'recall_macro'},
                                    return_train_score=False
                                )
                                accuracy_scores.extend(scores['test_acc'])
                                f1_scores.extend(scores['test_f1'])
                                recall_scores.extend(scores['test_rec'])
                            else:
                                # CV on eval data, always train on full main data
                                idxs = np.arange(len(X_eval))
                                for train_idx, test_idx in cv.split(idxs):
                                    X_eval_fold = X_eval.iloc[test_idx]
                                    y_eval_fold = y_eval[test_idx]

                                    classifier.random_state = seed_for_run
                                    classifier.fit(X_main, y_main)
                                    preds = classifier.predict(X_eval_fold)

                                    acc_ = accuracy_score(y_eval_fold, preds)
                                    f1_ = f1_score(y_eval_fold, preds, average='macro')
                                    rec_ = recall_score(y_eval_fold, preds, average='macro')

                                    accuracy_scores.append(acc_)
                                    f1_scores.append(f1_)
                                    recall_scores.append(rec_)

                    # Check for any ConvergenceWarning
                    for warning_msg in w:
                        if issubclass(warning_msg.category, ConvergenceWarning):
                            convergence_issues.add(clf_name)

                # best/worst/avg/std
                best_acc = max(accuracy_scores)
                worst_acc = min(accuracy_scores)
                avg_acc = np.mean(accuracy_scores)
                std_acc = np.std(accuracy_scores)

                best_f1 = max(f1_scores)
                worst_f1 = min(f1_scores)
                avg_f1 = np.mean(f1_scores)
                std_f1 = np.std(f1_scores)

                best_rec = max(recall_scores)
                worst_rec = min(recall_scores)
                avg_rec = np.mean(recall_scores)
                std_rec = np.std(recall_scores)

                self.results.append((
                    clf_name,
                    best_acc, worst_acc, avg_acc, std_acc,
                    best_f1, worst_f1, avg_f1, std_f1,
                    best_rec, worst_rec, avg_rec, std_rec
                ))

            if convergence_issues:
                messagebox.showinfo(
                    "Convergence Notice",
                    f"The following classifiers reached max_iter before converging:\n"
                    f"{', '.join(sorted(convergence_issues))}\n\n"
                    f"Consider increasing max_iter or adjusting other hyperparameters for these classifiers."
                )

            self.update_results_table()
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
        else:
            raise ValueError(f"Unknown classifier: {clf_name}")

    # ------------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------------
    def update_results_table(self):
        # Clear old rows
        for row in self.results_table.get_children():
            self.results_table.delete(row)

        for res in self.results:
            (
                clf_name,
                best_acc, worst_acc, avg_acc, std_acc,
                best_f1, worst_f1, avg_f1, std_f1,
                best_rec, worst_rec, avg_rec, std_rec
            ) = res

            row_values = (
                clf_name,
                f"{best_acc:.4f}",
                f"{worst_acc:.4f}",
                f"{avg_acc:.4f}",
                f"{std_acc:.4f}",
                f"{best_f1:.4f}",
                f"{worst_f1:.4f}",
                f"{avg_f1:.4f}",
                f"{std_f1:.4f}",
                f"{best_rec:.4f}",
                f"{worst_rec:.4f}",
                f"{avg_rec:.4f}",
                f"{std_rec:.4f}"
            )
            self.results_table.insert("", "end", values=row_values)

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
