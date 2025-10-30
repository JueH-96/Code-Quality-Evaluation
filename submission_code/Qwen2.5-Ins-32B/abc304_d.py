import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    W, H = int(data[idx]), int(data[idx+1])
    idx += 2
    
    N = int(data[idx])
    idx += 1
    
    strawberries = []
    for _ in range(N):
        p, q = int(data[idx]), int(data[idx+1])
        strawberries.append((p, q))
        idx += 2
    
    A = int(data[idx])
    idx += 1
    a = list(map(int, data[idx:idx+A]))
    idx += A
    
    B = int(data[idx])
    idx += 1
    b = list(map(int, data[idx:idx+B]))
    idx += B
    
    a = [0] + a + [W]
    b = [0] + b + [H]
    
    grid = [[0] * (len(b) - 1) for _ in range(len(a) - 1)]
    
    for p, q in strawberries:
        row = 0
        while a[row] < p:
            row += 1
        col = 0
        while b[col] < q:
            col += 1
        grid[row-1][col-1] += 1
    
    min_strawberries = min(min(row) for row in grid)
    max_strawberries = max(max(row) for row in grid)
    
    print(min_strawberries, max_strawberries)

if __name__ == "__main__":
    main()