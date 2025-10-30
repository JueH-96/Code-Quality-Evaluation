import sys

def solve():
    N = int(sys.stdin.readline().strip())
    P = list(map(int, sys.stdin.readline().strip().split()))
    M = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))

    inversions = 0
    for i in range(M):
        a = A[i]
        for j in range(i, M):
            if A[j] < a:
                inversions += 1

    print(inversions)

if __name__ == "__main__":
    solve()