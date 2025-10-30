# YOUR CODE HERE
import sys

def can_transform(N, K, A, B):
    from collections import defaultdict

    # Create a dictionary to store the positions of each value in A
    positions = defaultdict(list)
    for i, val in enumerate(A):
        positions[val].append(i)

    # Check if we can transform A to B
    for i in range(N):
        if A[i] != B[i]:
            found = False
            for j in positions[B[i]]:
                if abs(i - j) <= K:
                    found = True
                    break
            if not found:
                return "No"
    return "Yes"

T = int(input().strip())
for _ in range(T):
    N, K = map(int, input().strip().split())
    A = list(map(int, input().strip().split()))
    B = list(map(int, input().strip().split()))
    print(can_transform(N, K, A, B))