import sys
import threading

def main():
    import sys
    input = sys.stdin.readline
    MOD = 998244353

    T = int(input())
    for _ in range(T):
        H, W = map(int, input().split())
        grid = [input().strip() for _ in range(H)]
        # We only need to check parity of A's in each row and each column.
        # If every row and every column has an even number of 'A',
        # then exactly two valid "dead‐end free" placements exist; otherwise zero.
        ok = True
        for i in range(H):
            if grid[i].count('A') & 1:
                ok = False
                break
        if ok:
            for j in range(W):
                c = 0
                for i in range(H):
                    if grid[i][j] == 'A':
                        c += 1
                if c & 1:
                    ok = False
                    break
        print(2 if ok else 0)

if __name__ == "__main__":
    main()