from bisect import bisect_left, insort

N = int(input())
P = list(map(int, input().split()))
M = int(input())
A = list(map(int, input().split()))

# Initialize the BIT and the sorted list of elements
BIT = [0] * (N + 1)
sorted_elements = [0] * N

# Function to update the BIT
def update(idx, val):
    while idx <= N:
        BIT[idx] += val
        idx += idx & -idx

# Function to query the BIT
def query(idx):
    res = 0
    while idx > 0:
        res += BIT[idx]
        idx -= idx & -idx
    return res

# Initialize the sorted list with the permutation
for i, p in enumerate(P):
    sorted_elements[i] = p

# Sort the elements and update the BIT
sorted_elements.sort()
for i, p in enumerate(sorted_elements):
    update(p, 1)

# Initialize the inversion count
inversion_count = 0

# For each operation
for a in A:
    # Update the inversion count based on the operation
    for i in range(a - 1):
        if P[i] > P[i + 1]:
            # Find the index of P[i] and P[i + 1] in the sorted list
            idx1 = bisect_left(sorted_elements, P[i])
            idx2 = bisect_left(sorted_elements, P[i + 1])
            # Update the inversion count
            inversion_count -= query(P[i]) - query(P[i + 1])
            # Swap P[i] and P[i + 1]
            P[i], P[i + 1] = P[i + 1], P[i]
            # Update the sorted list and the BIT
            sorted_elements[idx1], sorted_elements[idx2] = sorted_elements[idx2], sorted_elements[idx1]
            update(P[i], -1)
            update(P[i + 1], 1)
    # Print the inversion count
    print(inversion_count)