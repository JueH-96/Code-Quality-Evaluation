import sys
from collections import defaultdict

n, X = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))

left = defaultdict(list)
right = defaultdict(list)

# Initialize right with elements from index 2 to n-1 (0-based)
for i in range(2, n):
    right[A[i]].append(i)

# Iterate j from 1 to n-2 (inclusive)
for j in range(1, n-1):
    # Add A[j-1] to left
    left[A[j-1]].append(j-1)
    # Compute target
    target = X - A[j]
    # Check all elements in left
    for a in left:
        b = target - a
        if b in right:
            # Find the first occurrence of a in left and b in right
            i = left[a][0]
            # Find the first occurrence of b in right which is > j
            for k in right[b]:
                if k > j:
                    print(i+1, j+1, k+1)
                    sys.exit()
    # Remove A[j+1] from right
    if j+1 < n:
        val = A[j+1]
        # Check if the first index in right[val] is j+1
        while right.get(val, []):
            if right[val][0] == j+1:
                right[val].pop(0)
                if not right[val]:
                    del right[val]
                break
            else:
                break

print(-1)