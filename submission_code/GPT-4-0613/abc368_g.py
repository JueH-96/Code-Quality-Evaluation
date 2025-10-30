import sys
from typing import List, Tuple

def solve(N: int, A: List[int], B: List[int], Q: int, queries: List[Tuple[int, ...]]) -> List[int]:
    for i in range(Q):
        if queries[i][0] == 1:
            A[queries[i][1]-1] = queries[i][2]
        elif queries[i][0] == 2:
            B[queries[i][1]-1] = queries[i][2]
        else:
            l, r = queries[i][1]-1, queries[i][2]
            v = 0
            for j in range(l, r):
                v = max(v + A[j], v * B[j])
            print(v)

def main():
    N = int(input().strip())
    A = list(map(int, input().strip().split()))
    B = list(map(int, input().strip().split()))
    Q = int(input().strip())
    queries = []
    for _ in range(Q):
        queries.append(tuple(map(int, input().strip().split())))
    solve(N, A, B, Q, queries)

if __name__ == "__main__":
    main()