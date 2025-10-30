n = int(input())
heights = list(map(int, input().split()))

result = []
for i in range(n):
    count = 0
    max_between = 0
    for j in range(i + 1, n):
        if j > i + 1:
            # Update max height between i and j by including the previous building
            max_between = max(max_between, heights[j-1])
        
        # If no building between i and j is taller than j, then j is visible
        if max_between <= heights[j]:
            count += 1
    
    result.append(count)

print(' '.join(map(str, result)))