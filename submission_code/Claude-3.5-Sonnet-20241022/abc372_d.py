N = int(input())
H = list(map(int, input().split()))

result = []
for i in range(N):
    count = 0
    max_height = 0
    for j in range(i+1, N):
        # Check if any building between i and j is taller than building j
        if max_height < H[j]:
            count += 1
        max_height = max(max_height, H[j])
    result.append(count)

print(*result)