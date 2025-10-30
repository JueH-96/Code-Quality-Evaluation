# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Find the second largest value
sorted_a = sorted(a, reverse=True)
second_largest = sorted_a[1]

# Find the position (1-indexed) of the second largest value
for i in range(n):
    if a[i] == second_largest:
        print(i + 1)  # Convert to 1-indexed
        break