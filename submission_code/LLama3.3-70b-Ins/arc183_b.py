import sys

def solve(N, K, A, B):
    for i in range(N):
        found = False
        for j in range(max(0, i-K), min(N, i+K+1)):
            if A[j] == B[i]:
                found = True
                break
        if not found:
            return "No"
    return "Yes"

T = int(input())
for _ in range(T):
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    print(solve(N, K, A, B))