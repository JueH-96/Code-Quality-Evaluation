from typing import List

def solve_case(N: int, A: List[int], B: List[int], C: List[int]) -> int:
    count = 0
    for x in range(1, 10**9+1):
        for y in range(1, 10**9+1):
            valid = True
            for i in range(N):
                if A[i] * x + B[i] * y >= C[i]:
                    valid = False
                    break
            if valid:
                count += 1
    return count

T = int(input())
for _ in range(T):
    N = int(input())
    A, B, C = [], [], []
    for _ in range(N):
        a, b, c = map(int, input().split())
        A.append(a)
        B.append(b)
        C.append(c)
    print(solve_case(N, A, B, C))