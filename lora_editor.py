# Script made by gemini, not my work

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import os
import re


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


class SafetensorsEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Safetensors Advanced Editor")
        self.root.geometry("1150x850")

        self.file_path = None
        self.tensors = {}
        self.metadata = {}
        self.pending_actions = {}

        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style()
        style.configure("Treeview", rowheight=25)

        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Button(top_frame, text="Load Safetensors", command=self.load_file).pack(side=tk.LEFT)

        search_container = tk.Frame(top_frame)
        search_container.pack(side=tk.LEFT, padx=20)

        tk.Label(search_container, text="Search:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *args: self.refresh_view())
        self.search_entry = tk.Entry(search_container, textvariable=self.search_var, width=40)
        self.search_entry.pack(side=tk.LEFT, padx=5)

        hint_label = tk.Label(top_frame, text="Hints: ^part (any block)  ^global.path  -neg", fg="#666",
                              font=("Arial", 8, "italic"))
        hint_label.pack(side=tk.LEFT)

        self.lbl_info = tk.Label(top_frame, text="No file loaded", fg="gray")
        self.lbl_info.pack(side=tk.RIGHT, padx=10)

        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        tree_container = tk.Frame(paned)
        tree_btns = tk.Frame(tree_container)
        tree_btns.pack(fill=tk.X)
        tk.Button(tree_btns, text="Expand All", command=self.expand_all, font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(tree_btns, text="Collapse All", command=self.collapse_all, font=('Arial', 8)).pack(side=tk.LEFT)

        self.tree = ttk.Treeview(tree_container, selectmode="extended")
        self.tree.heading("#0", text="Tensors / Blocks Hierarchy", anchor=tk.W)

        self.tree.tag_configure('delete', background='#ffcccb')
        self.tree.tag_configure('scale', background='#c8e6c9')
        self.tree.tag_configure('mixed', background='#fff9c4')

        tree_scroll = ttk.Scrollbar(tree_container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        paned.add(tree_container, stretch="always")

        ctrl_frame = tk.Frame(paned)
        paned.add(ctrl_frame, stretch="never")

        tk.Label(ctrl_frame, text="Actions on Selected", font=('Arial', 10, 'bold')).pack(pady=10, padx=20)

        tk.Label(ctrl_frame, text="Scale Factor:").pack()
        self.scale_entry = tk.Entry(ctrl_frame, justify='center')
        self.scale_entry.insert(0, "1.0")
        self.scale_entry.pack(pady=5, padx=20)

        tk.Button(ctrl_frame, text="Mark for Scaling", command=lambda: self.apply_action('scale'), bg="#e8f5e9").pack(
            fill=tk.X, pady=2, padx=20)
        tk.Button(ctrl_frame, text="Mark for Deletion", command=lambda: self.apply_action('delete'), bg="#ffebee").pack(
            fill=tk.X, pady=2, padx=20)
        tk.Button(ctrl_frame, text="Clear Actions", command=self.clear_actions).pack(fill=tk.X, pady=10, padx=20)

        legend_container = tk.Frame(ctrl_frame)
        legend_container.pack(side=tk.BOTTOM, pady=20)
        tk.Label(legend_container, text="Legend:", font=('Arial', 8, 'bold')).pack()
        tk.Label(legend_container, text="Red: All Modded | Yellow: Mixed", font=('Arial', 8)).pack()

        tk.Button(ctrl_frame, text="EXPORT MODIFIED", command=self.save_modified, bg="#4caf50", fg="white",
                  font=('Arial', 10, 'bold'), height=2).pack(side=tk.BOTTOM, fill=tk.X, pady=10, padx=20)

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Safetensors", "*.safetensors")])
        if not path: return
        try:
            with safe_open(path, framework="pt", device="cpu") as f:
                self.metadata = f.metadata()
                self.tensors = {k: f.get_tensor(k) for k in f.keys()}
            self.pending_actions = {}
            self.lbl_info.config(text=f"Loaded: {os.path.basename(path)}", fg="black")
            self.refresh_view()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def matches_query(self, full_key, query_parts):
        if not query_parts: return True

        full_key_lower = full_key.lower()
        key_segments = full_key_lower.split('.')

        for part in query_parts:
            # Negative search
            if part.startswith('-'):
                if part[1:] in full_key_lower: return False

            # Start anchor logic
            elif part.startswith('^'):
                pattern = part[1:]
                if '.' in pattern:
                    # Global path check: must start from the very beginning
                    if not full_key_lower.startswith(pattern): return False
                else:
                    # Component check: does ANY part of the path start with this pattern?
                    if not any(seg.startswith(pattern) for seg in key_segments): return False

            # End anchor logic
            elif part.endswith('$'):
                if not full_key_lower.endswith(part[:-1]): return False

            # Standard contains
            else:
                if part not in full_key_lower: return False
        return True

    def refresh_view(self):
        query_raw = self.search_var.get().lower()
        query_parts = [p for p in query_raw.split(' ') if p]

        for i in self.tree.get_children(): self.tree.delete(i)

        sorted_keys = sorted(self.tensors.keys(), key=natural_sort_key)
        for key in sorted_keys:
            if not self.matches_query(key, query_parts):
                continue

            parts = key.split('.')
            parent = ""
            for i, part in enumerate(parts):
                current_path = ".".join(parts[:i + 1])
                if not self.tree.exists(current_path):
                    self.tree.insert(parent, "end", text=part, iid=current_path, open=bool(query_parts))
                parent = current_path
        self.reapply_visuals()

    def expand_all(self):
        def _rec(n):
            self.tree.item(n, open=True)
            for c in self.tree.get_children(n): _rec(c)

        for r in self.tree.get_children(''): _rec(r)

    def collapse_all(self):
        def _rec(n):
            for c in self.tree.get_children(n): _rec(c)
            self.tree.item(n, open=False)

        for r in self.tree.get_children(''): _rec(r)

    def reapply_visuals(self):
        def clear_tags(n):
            self.tree.item(n, tags=())
            for c in self.tree.get_children(n): clear_tags(c)

        for root in self.tree.get_children(''): clear_tags(root)

        def evaluate(node):
            children = self.tree.get_children(node)
            if not children:
                if node in self.pending_actions:
                    action = self.pending_actions[node]['action']
                    self.tree.item(node, tags=(action,))
                    return action
                return "none"

            child_results = [evaluate(c) for c in children]
            unique = set(child_results)
            if len(unique) == 1:
                status = list(unique)[0]
                if status != "none": self.tree.item(node, tags=(status,))
                return status
            else:
                if any(r != "none" for r in child_results):
                    self.tree.item(node, tags=('mixed',))
                    return "mixed"
                return "none"

        for root in self.tree.get_children(''): evaluate(root)

    def get_visible_tensors_under(self, item_id):
        results = []
        if item_id in self.tensors: results.append(item_id)
        for child in self.tree.get_children(item_id):
            results.extend(self.get_visible_tensors_under(child))
        return results

    def apply_action(self, action_type):
        selected = self.tree.selection()
        val = 1.0
        if action_type == 'scale':
            try:
                val = float(self.scale_entry.get())
            except:
                return

        for item in selected:
            for t in self.get_visible_tensors_under(item):
                self.pending_actions[t] = {'action': action_type, 'value': val}
        self.reapply_visuals()

    def clear_actions(self):
        selected = self.tree.selection()
        for item in selected:
            for t in self.get_visible_tensors_under(item):
                if t in self.pending_actions: del self.pending_actions[t]
        self.reapply_visuals()

    def save_modified(self):
        if not self.tensors: return
        save_path = filedialog.asksaveasfilename(defaultextension=".safetensors")
        if not save_path: return
        out = {}
        for k, v in self.tensors.items():
            act = self.pending_actions.get(k)
            if act:
                if act['action'] == 'delete': continue
                if act['action'] == 'scale': out[k] = v * act['value']
            else:
                out[k] = v
        try:
            save_file(out, save_path, metadata=self.metadata)
            messagebox.showinfo("Done", "Exported successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = SafetensorsEditor(root)
    root.mainloop()