import sys
from collections import defaultdict

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # Check if the conditions are satisfied
    if len(set(A)) != N or any(b != -1 and b in A for b in B):
        return 0

    # Create a dictionary to store the indices of each element in A
    a_indices = defaultdict(list)
    for i, a in enumerate(A):
        a_indices[a].append(i)

    # Create a dictionary to store the indices of each element in B
    b_indices = defaultdict(list)
    for i, b in enumerate(B):
        if b != -1:
            b_indices[b].append(i)

    # Count the number of valid pairs
    count = 1
    for i in range(1, 2*N+1):
        if i in a_indices and i in b_indices:
            count = (count * len(a_indices[i]) * len(b_indices[i])) % 998244353

    return count

print(solve())