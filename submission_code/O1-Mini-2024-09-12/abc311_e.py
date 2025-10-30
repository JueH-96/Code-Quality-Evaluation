# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    H, W, N = map(int, sys.stdin.readline().split())
    holed_rows = [set() for _ in range(H+1)]
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        holed_rows[a].add(b)
    prev = [0] * (W+1)
    current = [0] * (W+1)
    total = 0
    for i in range(1, H+1):
        row_holes = holed_rows[i]
        for j in range(1, W+1):
            if j in row_holes:
                current[j] = 0
            else:
                current[j] = 1 + min(prev[j], current[j-1], prev[j-1])
            total += current[j]
        prev, current = current, [0]*(W+1)
    print(total)

if __name__ == "__main__":
    main()