import tkinter as tk
from tkinter import ttk

# Dữ liệu đã loại bỏ "Triangle_Ascending" và đánh lại số thứ tự từ 1-14
TABLE_DATA = [
    (1, "Diamond_Top", "1h", "EMA Peaks", "🔻 Đảo -> Long", "55.90%", "1960", "🟢 Cao"),
    (2, "Diamond_Top", "1h", "Pivot Window", "🔻 Đảo -> Long", "57.63%", "1960", "🟢 Cao"),
    (3, "Broadening_Bottom", "1h", "Pivot Window", "➡️ Up (breakout)", "55.54%", "632", "🟡 Trung bình"),
    (4, "Triple_Top_Wide (mới)", "1h", "ZigZag Limit", "➡️ Short", "60.37%", "540", "🟢 Cao"),
    (5, "Wedge_Falling", "15m", "ZigZag Limit", "➡️ Long", "56.88%", "2187", "🟢 Cao"),
    (6, "Triple_Bottom_Wide (mới)", "15m", "ZigZag Limit", "➡️ Long", "62.34%", "632", "🟢 Cao"),
    (7, "Cup_and_Handle", "15m", "ZigZag Limit", "🔻 Đảo -> Short", "56.98%", "946", "🟢 Cao"),
    (8, "Double_Bottom", "15m", "ZigZag Limit", "🔻 Đảo -> Short", "56.03%", "1508", "🟢 Cao"),
    (9, "Triple_Top", "15m", "ZigZag Limit", "➡️ Short", "58.49%", "265", "🟡 Trung bình"),
    (10, "Head_and_Shoulders_Top", "15m", "ZigZag Limit", "➡️ Short", "55.75%", "400", "🟡 Trung bình"),
    (11, "Head_and_Shoulders_Bottom", "15m", "ZigZag Limit", "➡️ Long (giữ nguyên)", "57.81%", "704", "🟢 Cao"),
    (12, "Pennant_Bearish", "15m", "Pivot Window", "🔻 Đảo -> Long", "55.46%", "1549", "🟢 Cao"),
    (13, "Broadening_Bottom", "4h", "ZigZag Limit", "➡️ Up (breakout)", "56.38%", "315", "🟡 Trung bình"),
    (14, "Wedge_Rising", "4h", "Pivot Window", "➡️ Short", "66.07%", "120", "🟢 Cao"),
]

def create_ui():
    root = tk.Tk()
    root.title("Bộ pattern hoàn thiện")
    root.geometry("1000x650")
    root.configure(bg="white")

    # --- STYLE TÙY CHỈNH (Custom Style) ---
    style = ttk.Style()
    style.theme_use("clam")  

    style.configure(
        "Custom.Treeview",
        background="white",
        foreground="#333333",
        rowheight=40,            
        fieldbackground="white",
        borderwidth=0,
        font=("Segoe UI", 10)
    )
    
    style.configure(
        "Custom.Treeview.Heading",
        background="#f8f9fa",    
        foreground="#111827",
        font=("Segoe UI", 10, "bold"),
        borderwidth=0,
        relief="flat"
    )
    
    style.map("Custom.Treeview", background=[("selected", "#e5e7eb")], foreground=[("selected", "black")])

    # --- TIÊU ĐỀ ---
    header_label = tk.Label(
        root, 
        text="Bộ pattern hoàn thiện (14 pattern)", 
        font=("Segoe UI", 16, "bold"), 
        bg="white",
        anchor="w"
    )
    header_label.pack(fill="x", padx=20, pady=(20, 10))

    # --- KHỞI TẠO BẢNG (Treeview) ---
    columns = ("stt", "pattern", "tf", "method", "direction", "wr", "n", "trust")
    tree = ttk.Treeview(root, columns=columns, show="headings", style="Custom.Treeview")

    tree.heading("stt", text="#")
    tree.heading("pattern", text="Pattern")
    tree.heading("tf", text="TF")
    tree.heading("method", text="Phương pháp")
    tree.heading("direction", text="Hướng vào lệnh")
    tree.heading("wr", text="Win rate")
    tree.heading("n", text="n")
    tree.heading("trust", text="Độ tin cậy")

    tree.column("stt", width=40, anchor="center")
    tree.column("pattern", width=200, anchor="w")
    tree.column("tf", width=50, anchor="center")
    tree.column("method", width=150, anchor="w")
    tree.column("direction", width=180, anchor="w")
    tree.column("wr", width=80, anchor="center")
    tree.column("n", width=60, anchor="center")
    tree.column("trust", width=120, anchor="w")

    # --- CHÈN DỮ LIỆU VÀO BẢNG ---
    for i, row in enumerate(TABLE_DATA):
        tag = "even" if i % 2 == 0 else "odd"
        tree.insert("", "end", values=row, tags=(tag,))

    tree.tag_configure("even", background="#ffffff")
    tree.tag_configure("odd", background="#fafafa")

    tree.pack(expand=True, fill="both", padx=20, pady=(0, 20))

    root.mainloop()

if __name__ == "__main__":
    create_ui()