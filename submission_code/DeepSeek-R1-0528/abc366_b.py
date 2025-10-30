def main():
    n = int(input().strip())
    strings = []
    for _ in range(n):
        strings.append(input().strip())
    
    if n == 0:
        return
    
    m = max(len(s) for s in strings)
    grid = [['*'] * n for _ in range(m)]
    
    for idx, s in enumerate(strings):
        col = n - 1 - idx
        for j in range(len(s)):
            grid[j][col] = s[j]
    
    for row in grid:
        s = ''.join(row)
        s_clean = s.rstrip('*')
        print(s_clean)

if __name__ == "__main__":
    main()