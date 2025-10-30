# YOUR CODE HERE
import sys

def solve(n, x, k):
    if k == 0:
        return 1
    if k > n - x:
        return 0
    if x % 2 == 0:
        return solve(n, x // 2, k - 1) + solve(n, (x + 1) // 2, k - 1)
    else:
        return solve(n, x // 2, k - 1)

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n, x, k = map(int, sys.stdin.readline().split())
        print(solve(n, x, k))

if __name__ == "__main__":
    main()