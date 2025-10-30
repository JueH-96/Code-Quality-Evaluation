# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Get unique values and sort them
unique_values = sorted(set(a), reverse=True)

# The second element is the answer (largest among those that are not the largest)
print(unique_values[1])