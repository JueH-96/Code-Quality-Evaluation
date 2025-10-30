import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    s_grid = [sys.stdin.readline().strip() for _ in range(n)]
    t_grid = [sys.stdin.readline().strip() for _ in range(m)]
    
    for start_row in range(n - m + 1):
        for start_col in range(n - m + 1):
            match = True
            for i in range(m):
                for j in range(m):
                    if s_grid[start_row + i][start_col + j] != t_grid[i][j]:
                        match = False
                        break
                if not match:
                    break
            if match:
                print(start_row + 1, start_col + 1)
                return

if __name__ == "__main__":
    main()