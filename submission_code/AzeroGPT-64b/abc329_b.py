n = int(input())
A = list(map(int, input().split()))

# Find the largest number in the list
largest = max(A)

# Remove all occurrences of the largest number from the list
A=[i for i in A if i != largest]

# The largest number in the remaining list is the answer
print(max(A))