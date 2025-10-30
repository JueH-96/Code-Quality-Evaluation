import sys

def main():
    n = int(sys.stdin.readline())
    grid_a = [sys.stdin.readline().strip() for _ in range(n)]
    grid_b = [sys.stdin.readline().strip() for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if grid_a[i][j] != grid_b[i][j]:
                print(f"{i+1} {j+1}")
                return

if __name__ == "__main__":
    main()