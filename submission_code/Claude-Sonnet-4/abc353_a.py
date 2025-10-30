# YOUR CODE HERE
n = int(input())
heights = list(map(int, input().split()))

first_height = heights[0]
result = -1

for i in range(1, n):
    if heights[i] > first_height:
        result = i + 1  # Convert to 1-indexed position
        break

print(result)