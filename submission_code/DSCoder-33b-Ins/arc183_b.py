# YOUR CODE HERE

import sys

def solve():
    T = int(input())
    for _ in range(T):
        N, K = map(int, input().split())
        A = list(map(int, input().split()))
        B = list(map(int, input().split()))

        # Create a dictionary to store the indices of each number in A
        indices = {}
        for i in range(N):
            if A[i] not in indices:
                indices[A[i]] = []
            indices[A[i]].append(i)

        # Check if it is possible to make A identical to B
        for i in range(N):
            if B[i] not in indices:
                print("No")
                break
            found = False
            for j in indices[B[i]]:
                if abs(i - j) <= K:
                    found = True
                    break
            if not found:
                print("No")
                break
        else:
            print("Yes")

solve()