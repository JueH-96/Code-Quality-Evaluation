import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    A.sort()
    ans = float('inf')
    for i in range(K + 1):
        B = A[i:N - (K - i)]
        ans = min(ans, max(B) - min(B))
    print(ans)

if __name__ == "__main__":
    solve()