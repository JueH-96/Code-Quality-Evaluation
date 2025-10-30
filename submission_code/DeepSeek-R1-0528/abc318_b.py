def main():
    n = int(input().strip())
    grid = [[False] * 100 for _ in range(100)]
    
    for _ in range(n):
        a, b, c, d = map(int, input().split())
        for x in range(a, b):
            for y in range(c, d):
                grid[x][y] = True
                
    total_area = 0
    for i in range(100):
        for j in range(100):
            if grid[i][j]:
                total_area += 1
                
    print(total_area)

if __name__ == "__main__":
    main()