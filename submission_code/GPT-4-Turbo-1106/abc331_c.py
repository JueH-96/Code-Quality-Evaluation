# Read the number of elements
N = int(input())

# Read the elements of the sequence A
A = list(map(int, input().split()))

# Calculate the total sum of the elements in A
total_sum = sum(A)

# Initialize an empty list to store the results
B = []

# Calculate the sum of elements greater than A_i for each i
for i in range(N):
    # Subtract A_i from the total sum to get the sum of elements greater than A_i
    B.append(total_sum - A[i] - sum(x <= A[i] for x in A) * A[i])

# Print the results separated by spaces
print(' '.join(map(str, B)))