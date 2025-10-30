import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    A.sort()

    max_gifts = 0
    for i in range(N):
        gifts = 0
        for j in range(i, N):
            if A[j] < A[i] + M:
                gifts += 1
            else:
                break
        max_gifts = max(max_gifts, gifts)

    print(max_gifts)

if __name__ == "__main__":
    solve()