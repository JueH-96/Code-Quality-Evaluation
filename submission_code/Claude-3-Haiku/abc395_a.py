# Read the input
N = int(input())
A = list(map(int, input().split()))

# Check if the sequence is strictly increasing
is_strictly_increasing = all(A[i] < A[i+1] for i in range(len(A)-1))

# Print the result
print("Yes" if is_strictly_increasing else "No")