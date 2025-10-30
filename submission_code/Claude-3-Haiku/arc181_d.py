from typing import List

def inversion_count(arr: List[int]) -> int:
    count = 0
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] > arr[j]:
                count += 1
    return count

def solve():
    N = int(input())
    P = list(map(int, input().split()))
    M = int(input())
    A = list(map(int, input().split()))

    for i in range(M):
        for j in range(A[i]-1):
            if P[j] > P[j+1]:
                P[j], P[j+1] = P[j+1], P[j]
        print(inversion_count(P))

solve()