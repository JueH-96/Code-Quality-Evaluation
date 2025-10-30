import sys

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    S = []
    T = []
    for _ in range(N-1):
        s, t = map(int, input().split())
        S.append(s)
        T.append(t)

    for i in range(N-2, -1, -1):
        if A[i] >= S[i]:
            num_operations = A[i] // S[i]
            A[i] -= num_operations * S[i]
            A[i+1] += num_operations * T[i]

    print(A[-1])

if __name__ == "__main__":
    solve()