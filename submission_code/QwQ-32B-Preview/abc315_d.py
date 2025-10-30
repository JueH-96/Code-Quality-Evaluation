from collections import deque

def main():
    import sys
    from sys import stdin
    input = stdin.read().splitlines()
    
    H_W = input[0].split()
    H = int(H_W[0])
    W = int(H_W[1])
    grid = [list(input[i]) for i in range(1, H+1)]
    
    present = [[True for _ in range(W)] for _ in range(H)]
    row_counts = [{} for _ in range(H)]
    col_counts = [{} for _ in range(W)]
    row_present_counts = [W for _ in range(H)]
    col_present_counts = [H for _ in range(W)]
    
    for i in range(H):
        for j in range(W):
            c = grid[i][j]
            row_counts[i][c] = row_counts[i].get(c, 0) + 1
            col_counts[j][c] = col_counts[j].get(c, 0) + 1
    
    queue = deque()
    for i in range(H):
        queue.append((0, i))  # 0 indicates row
    
    while queue:
        item = queue.popleft()
        if item[0] == 0:  # row
            i = item[1]
            if row_present_counts[i] >= 2:
                max_color = None
                max_count = 0
                for c, count in row_counts[i].items():
                    if count > max_count:
                        max_color = c
                        max_count = count
                if max_count == row_present_counts[i]:
                    for j in range(W):
                        if present[i][j]:
                            present[i][j] = False
                            row_present_counts[i] -= 1
                            col_present_counts[j] -= 1
                            c = grid[i][j]
                            col_counts[j][c] -= 1
                            if col_counts[j][c] == 0:
                                del col_counts[j][c]
                            queue.append((1, j))
        else:  # column
            j = item[1]
            if col_present_counts[j] >= 2:
                max_color = None
                max_count = 0
                for c, count in col_counts[j].items():
                    if count > max_count:
                        max_color = c
                        max_count = count
                if max_count == col_present_counts[j]:
                    for i in range(H):
                        if present[i][j]:
                            present[i][j] = False
                            row_present_counts[i] -= 1
                            col_present_counts[j] -= 1
                            c = grid[i][j]
                            row_counts[i][c] -= 1
                            if row_counts[i][c] == 0:
                                del row_counts[i][c]
                            queue.append((0, i))
    
    remaining = sum(present[i][j] for i in range(H) for j in range(W))
    print(remaining)

if __name__ == '__main__':
    main()