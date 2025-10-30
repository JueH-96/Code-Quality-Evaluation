# YOUR CODE HERE
n = int(input())
heights = list(map(int, input().split()))

result = []

for i in range(n):
    count = 0
    max_height = 0
    
    for j in range(i + 1, n):
        if heights[j] >= max_height:
            count += 1
        max_height = max(max_height, heights[j])
    
    result.append(count)

print(' '.join(map(str, result)))