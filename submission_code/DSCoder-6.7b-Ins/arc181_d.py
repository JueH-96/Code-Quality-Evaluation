import sys

def read_input():
    N = int(sys.stdin.readline().strip())
    P = list(map(int, sys.stdin.readline().strip().split()))
    M = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))
    return N, P, M, A

def count_inversions(P):
    inversions = 0
    for i in range(len(P)):
        for j in range(i+1, len(P)):
            if P[i] > P[j]:
                inversions += 1
    return inversions

def solve():
    N, P, M, A = read_input()
    inversions = [0] * M
    for i in range(M):
        k = A[i]
        for j in range(k-1, -1, -1):
            if P[j] > P[j+1]:
                P[j], P[j+1] = P[j+1], P[j]
                inversions[i] += 1
    return inversions

if __name__ == "__main__":
    inversions = solve()
    for inv in inversions:
        print(inv)