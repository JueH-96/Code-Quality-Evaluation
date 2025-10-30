# YOUR CODE HERE
def count_inversions(arr):
    count = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                count += 1
    return count

def perform_operation(P, k):
    # Operation k: bubble sort pass on first k elements
    for i in range(k - 1):
        if P[i] > P[i + 1]:
            P[i], P[i + 1] = P[i + 1], P[i]

# Read input
N = int(input())
P = list(map(int, input().split()))
M = int(input())
A = list(map(int, input().split()))

# Process each operation
for i in range(M):
    perform_operation(P, A[i])
    # Count inversions
    inversions = count_inversions(P)
    print(inversions)