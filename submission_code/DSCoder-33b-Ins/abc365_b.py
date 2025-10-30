# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

# Sort the list in descending order
A.sort(reverse=True)

# Find the second largest element
second_largest = A[1]

# Find the index of the second largest element
index = A.index(second_largest) + 1

print(index)