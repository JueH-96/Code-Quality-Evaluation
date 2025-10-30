n = int(input())
heights = list(map(int, input().split()))

result = []

for i in range(n):
    count = 0
    max_height_between = 0
    for j in range(i + 1, n):
        if max_height_between <= heights[j]:
            count += 1
        max_height_between = max(max_height_between, heights[j])
    result.append(count)

print(' '.join(map(str, result)))