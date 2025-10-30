# Read the input
N = int(input())
A = list(map(int, input().split()))

# Find the second largest element
A.sort(reverse=True)
second_largest = A[1]

# Print the index of the second largest element
print(A.index(second_largest) + 1)