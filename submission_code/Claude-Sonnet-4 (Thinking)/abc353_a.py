n = int(input())
heights = list(map(int, input().split()))

first_height = heights[0]

for i in range(1, n):
    if heights[i] > first_height:
        print(i + 1)  # Convert to 1-indexed position
        break
else:
    print(-1)