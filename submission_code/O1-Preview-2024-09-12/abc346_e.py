# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    H, W, M = map(int, sys.stdin.readline().split())
    operations = []
    for _ in range(M):
        T_i, A_i, X_i = sys.stdin.readline().split()
        T_i = int(T_i)
        A_i = int(A_i)
        X_i = int(X_i)
        operations.append((T_i, A_i, X_i))
    processed_rows = set()
    processed_columns = set()
    color_counts = {}
    total_cells = 0
    for op in reversed(operations):
        T_i, A_i, X_i = op
        if T_i == 1:
            # Row operation
            if A_i not in processed_rows:
                num_cells = W - len(processed_columns)
                color_counts[X_i] = color_counts.get(X_i, 0) + num_cells
                total_cells += num_cells
                processed_rows.add(A_i)
        elif T_i ==2:
            # Column operation
            if A_i not in processed_columns:
                num_cells = H - len(processed_rows)
                color_counts[X_i] = color_counts.get(X_i, 0) + num_cells
                total_cells += num_cells
                processed_columns.add(A_i)
    unpainted_cells = H * W - total_cells
    if unpainted_cells > 0:
        color_counts[0] = color_counts.get(0, 0) + unpainted_cells
    K = len(color_counts)
    print(K)
    for color in sorted(color_counts.keys()):
        count = color_counts[color]
        if count > 0:
            print(f"{color} {count}")

threading.Thread(target=main).start()