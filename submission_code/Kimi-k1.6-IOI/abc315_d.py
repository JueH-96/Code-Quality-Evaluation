def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H = int(data[idx])
    idx += 1
    W = int(data[idx])
    idx += 1
    grid = []
    for _ in range(H):
        grid.append(list(data[idx]))
        idx += 1
    
    present = [[True for _ in range(W)] for _ in range(H)]
    
    row_count = [W] * H
    row_freq = [[0] * 26 for _ in range(H)]
    max_row_freq = [0] * H
    for i in range(H):
        freq = [0] * 26
        for c in grid[i]:
            freq[ord(c) - ord('a')] += 1
        row_freq[i] = freq
        max_row_freq[i] = max(freq)
    
    col_count = [H] * W
    col_freq = [[0] * 26 for _ in range(W)]
    max_col_freq = [0] * W
    for j in range(W):
        freq = [0] * 26
        for i in range(H):
            freq[ord(grid[i][j]) - ord('a')] += 1
        col_freq[j] = freq
        max_col_freq[j] = max(freq)
    
    while True:
        R = [i for i in range(H) if row_count[i] >= 2 and max_row_freq[i] == row_count[i]]
        C = [j for j in range(W) if col_count[j] >= 2 and max_col_freq[j] == col_count[j]]
        
        if not R and not C:
            break
        
        to_remove = set()
        for i in R:
            for j in range(W):
                if present[i][j]:
                    to_remove.add((i, j))
        for j in C:
            for i in range(H):
                if present[i][j]:
                    to_remove.add((i, j))
        
        if not to_remove:
            break
        
        for i, j in to_remove:
            if not present[i][j]:
                continue
            present[i][j] = False
            c = grid[i][j]
            idx_char = ord(c) - ord('a')
            
            row_count[i] -= 1
            row_freq[i][idx_char] -= 1
            max_f = 0
            for k in range(26):
                if row_freq[i][k] > max_f:
                    max_f = row_freq[i][k]
            max_row_freq[i] = max_f
            
            col_count[j] -= 1
            col_freq[j][idx_char] -= 1
            max_f_col = 0
            for k in range(26):
                if col_freq[j][k] > max_f_col:
                    max_f_col = col_freq[j][k]
            max_col_freq[j] = max_f_col
    
    count = 0
    for row in present:
        count += sum(row)
    print(count)

if __name__ == '__main__':
    main()