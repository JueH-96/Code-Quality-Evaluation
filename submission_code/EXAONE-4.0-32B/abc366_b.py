import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n = int(data[0])
    strings = data[1:1+n]
    
    if n == 0:
        return
        
    M = max(len(s) for s in strings)
    
    grid = [[None] * M for _ in range(n)]
    
    for i, s in enumerate(strings):
        row_idx = n - 1 - i
        for j, char in enumerate(s):
            if j < M:
                grid[row_idx][j] = char
                
    for j in range(M):
        col_chars = []
        for i in range(n):
            if grid[i][j] is None:
                col_chars.append('*')
            else:
                col_chars.append(grid[i][j])
        s_col = ''.join(col_chars)
        s_col_stripped = s_col.rstrip('*')
        print(s_col_stripped)

if __name__ == "__main__":
    main()