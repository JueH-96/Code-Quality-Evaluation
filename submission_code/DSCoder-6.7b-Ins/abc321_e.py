import sys

def solve():
    T = int(sys.stdin.readline().strip())
    for _ in range(T):
        N, X, K = map(int, sys.stdin.readline().strip().split())
        count = 0
        for i in range(X, N+1, 2):
            if i // 2 + K == X:
                count += 1
        print(count)

if __name__ == "__main__":
    solve()