# YOUR CODE HERE
N = int(input())
heights = list(map(int, input().split()))

first_height = heights[0]
result = -1

for i in range(1, N):
    if heights[i] > first_height:
        result = i + 1
        break

print(result)