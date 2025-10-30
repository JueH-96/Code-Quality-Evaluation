# Read the number of elements in the sequence
N = int(input())

# Read the sequence
A = list(map(int, input().split()))

# Sort the sequence in descending order
sorted_A = sorted(A, reverse=True)

# Find the second largest element
second_largest = sorted_A[1]

# Find the index of the second largest element in the original sequence
index = A.index(second_largest) + 1

# Print the index of the second largest element
print(index)