# Read the number of elements
N = int(input())

# Read the elements and convert them to a list of tuples (element, index)
A = list(map(int, input().split()))
indexed_A = [(value, index + 1) for index, value in enumerate(A)]

# Sort the list by the elements in descending order
sorted_A = sorted(indexed_A, key=lambda x: x[0], reverse=True)

# The second element in the sorted list is the second largest, print its original index
print(sorted_A[1][1])