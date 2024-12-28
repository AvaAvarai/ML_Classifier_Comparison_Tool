import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from tkinter import ttk
import random

class ClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Classifier GUI")

        self.file_label = tk.Label(root, text="No file loaded")
        self.file_label.pack()

        self.load_button = tk.Button(root, text="Load File", command=self.load_file)
        self.load_button.pack()

        self.info_text = tk.Text(root, height=10, state=tk.DISABLED)
        self.info_text.pack()

        self.setup_classifier_selection()
        self.setup_run_parameters()
        self.setup_run_buttons()

        self.results_table = ttk.Treeview(root, columns=("Classifier", "Best", "Worst", "Average", "Std Dev"), show="headings")
        for col in self.results_table['columns']:
            self.results_table.heading(col, text=col)
        self.results_table.pack()

        self.export_button = tk.Button(root, text="Export to CSV", command=self.export_results)
        self.export_button.pack()

        self.data = None
        self.class_column = None
        self.results = []

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
        balance_ratio = class_counts.min() / class_counts.max()

        info = (
            f"Classes: {list(class_counts.index)}\n"
            f"Class Counts: {class_counts.to_dict()}\n"
            f"Total Cases: {total_cases}\n"
            f"Number of Classes: {num_classes}\n"
            f"Class Balance: {'Balanced' if balance_ratio == 1 else 'Not Balanced'} (Ratio: {balance_ratio:.2f})\n"
            f"Number of Features: {len(self.data.columns) - 1}\n"  # -1 to exclude class column
            f"Features: {[col for col in self.data.columns if col != self.class_column]}\n"
            f"Missing Values: {'none' if self.data.isnull().sum().sum() == 0 else f'contains missing values ({self.data.isnull().sum().sum()})'}"
        )

        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info)
        self.info_text.config(state=tk.DISABLED)

    def setup_classifier_selection(self):
        self.classifiers_frame = tk.LabelFrame(self.root, text="Classifiers")
        self.classifiers_frame.pack(fill=tk.X)

        self.selected_classifiers = {}

        self.add_classifier_checkbox("KNN", self.classifiers_frame)
        self.add_classifier_checkbox("SVM", self.classifiers_frame)
        self.add_classifier_checkbox("Logistic Regression", self.classifiers_frame)
        self.add_classifier_checkbox("Naive Bayes", self.classifiers_frame)
        self.add_classifier_checkbox("Decision Tree", self.classifiers_frame)
        self.add_classifier_checkbox("Random Forest", self.classifiers_frame)
        self.add_classifier_checkbox("AdaBoost", self.classifiers_frame)
        self.add_classifier_checkbox("Extra Trees", self.classifiers_frame)

    def add_classifier_checkbox(self, name, frame):
        var = tk.BooleanVar(value=False)
        chk = tk.Checkbutton(frame, text=name, variable=var)
        chk.pack(side=tk.LEFT)
        self.selected_classifiers[name] = var

    def setup_run_parameters(self):
        self.params_frame = tk.LabelFrame(self.root, text="Parameters")
        self.params_frame.pack(fill=tk.X)

        # Add parameter entries with validation
        self.k_value = self.add_param_entry("K Value (KNN):", self.params_frame, "5")
        self.svm_kernel = self.add_param_entry("SVM Kernel (linear/rbf):", self.params_frame, "rbf")
        self.cross_val_split = self.add_param_entry("Cross-Validation Split:", self.params_frame, "5")
        self.run_count = self.add_param_entry("Run Count:", self.params_frame, "10")
        self.random_seed = self.add_param_entry("Random Seed:", self.params_frame, str(random.randint(0, 10**6)))

    def add_param_entry(self, label, frame, default):
        lbl = tk.Label(frame, text=label)
        lbl.pack(side=tk.LEFT)
        entry = tk.Entry(frame)
        entry.insert(0, default)
        entry.pack(side=tk.LEFT)
        return entry

    def setup_run_buttons(self):
        self.run_button = tk.Button(self.root, text="Run", command=self.run_classifiers)
        self.run_button.pack()

    def run_classifiers(self):
        if self.data is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return

        try:
            random_seed = int(self.random_seed.get())
            k_value = int(self.k_value.get())
            svm_kernel = self.svm_kernel.get().lower()
            if svm_kernel not in ['linear', 'rbf']:
                raise ValueError("SVM kernel must be 'linear' or 'rbf'")
            
            classifiers = {
                "KNN": KNeighborsClassifier(n_neighbors=k_value),
                "SVM": SVC(kernel=svm_kernel, random_state=random_seed),
                "Logistic Regression": LogisticRegression(random_state=random_seed, max_iter=1000),
                "Naive Bayes": GaussianNB(),
                "Decision Tree": DecisionTreeClassifier(random_state=random_seed),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_seed),
                "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=random_seed),
                "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=random_seed)
            }

            num_splits = int(self.cross_val_split.get())
            if num_splits < 2:
                raise ValueError("Cross-validation splits must be at least 2")
                
            run_count = int(self.run_count.get())
            if run_count < 1:
                raise ValueError("Run count must be at least 1")

            self.results = []

            for name, var in self.selected_classifiers.items():
                if not var.get():
                    continue

                accuracies = []

                for run in range(run_count):
                    # Set seeds for reproducibility
                    random.seed(random_seed + run)
                    np.random.seed(random_seed + run)
                    
                    # Create stratified k-fold cross validation
                    cv = KFold(n_splits=num_splits, shuffle=True, random_state=random_seed + run)
                    
                    # Get features and target
                    X = self.data.drop(columns=self.class_column)
                    y = self.data[self.class_column]
                    
                    # Perform cross validation
                    scores = cross_val_score(classifiers[name], X, y, cv=cv, scoring='accuracy')
                    accuracies.extend(scores)

                best = max(accuracies)
                worst = min(accuracies)
                average = np.mean(accuracies)
                std_dev = np.std(accuracies)

                self.results.append((name, best, worst, average, std_dev))

            self.update_results_table()
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def update_results_table(self):
        for row in self.results_table.get_children():
            self.results_table.delete(row)

        for result in self.results:
            formatted_result = (
                result[0],
                f"{result[1]:.4f}",
                f"{result[2]:.4f}", 
                f"{result[3]:.4f}",
                f"{result[4]:.4f}"
            )
            self.results_table.insert("", "end", values=formatted_result)

    def export_results(self):
        if not self.results:
            messagebox.showerror("Error", "No results to export.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        df = pd.DataFrame(self.results, columns=["Classifier", "Best", "Worst", "Average", "Std Dev"])
        df.to_csv(file_path, index=False)
        messagebox.showinfo("Success", "Results exported successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ClassifierApp(root)
    root.mainloop()
