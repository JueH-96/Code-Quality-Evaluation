# Read input
n = int(input())
a = list(map(int, input().split()))

# Create a list of tuples (value, original index)
pairs = [(num, idx + 1) for idx, num in enumerate(a)]

# Sort the list in descending order of values
sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

# The second element in the sorted list is the second largest
# Output its original index
print(sorted_pairs[1][1])