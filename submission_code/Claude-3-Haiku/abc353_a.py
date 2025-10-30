# Read the input
N = int(input())
heights = list(map(int, input().split()))

# Find the leftmost building taller than the first one
first_height = heights[0]
for i, height in enumerate(heights):
    if height > first_height:
        print(i+1)
        break
else:
    print(-1)