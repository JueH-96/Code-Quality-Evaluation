# YOUR CODE HERE
n = int(input())
heights = list(map(int, input().split()))

result = []

for i in range(n):
    count = 0
    for j in range(i + 1, n):
        # Check if there's any building between i and j that's taller than building j
        max_height_between = 0
        for k in range(i + 1, j):
            max_height_between = max(max_height_between, heights[k])
        
        # If no building between i and j is taller than building j, count it
        if max_height_between < heights[j]:
            count += 1
    
    result.append(count)

print(' '.join(map(str, result)))