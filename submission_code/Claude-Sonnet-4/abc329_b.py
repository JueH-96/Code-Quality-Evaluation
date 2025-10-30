# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Find the largest value
max_val = max(a)

# Find all values that are not the largest
not_largest = [x for x in a if x != max_val]

# Find the largest among those that are not the largest
answer = max(not_largest)

print(answer)