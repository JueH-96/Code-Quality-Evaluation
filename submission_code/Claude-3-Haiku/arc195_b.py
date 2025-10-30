import sys

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # Check if it's possible to make all elements non-negative
    if any(a == -1 for a in A) or any(b == -1 for b in B):
        print("Yes")
        return

    # Check if the sum of corresponding elements is equal
    target_sum = A[0] + B[0]
    for i in range(1, N):
        if A[i] + B[i] != target_sum:
            print("No")
            return

    print("Yes")

solve()