N = int(input())
A = list(map(int, input().split()))

# Create list of tuples with (value, index)
indexed_A = list(enumerate(A, 1))

# Sort by value in descending order
sorted_A = sorted(indexed_A, key=lambda x: x[1], reverse=True)

# Print index of second largest element
print(sorted_A[1][0])