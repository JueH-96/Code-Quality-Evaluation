import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    A = list(map(int, data[2:]))
    
    grid = [[0] * N for _ in range(N)]
    row_count = [0] * N
    col_count = [0] * N
    diag_count = [0] * 2
    
    for turn, num in enumerate(A, 1):
        i, j = divmod(num - 1, N)
        grid[i][j] = 1
        
        row_count[i] += 1
        col_count[j] += 1
        
        if i == j:
            diag_count[0] += 1
        if i + j == N - 1:
            diag_count[1] += 1
        
        if row_count[i] == N or col_count[j] == N or diag_count[0] == N or diag_count[1] == N:
            print(turn)
            return
    
    print(-1)

if __name__ == "__main__":
    main()