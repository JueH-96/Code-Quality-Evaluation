n = int(input())
heights = list(map(int, input().split()))

max_count = 1  # At least we can choose 1 building

# Try all pairs of buildings as potential start of arithmetic sequence
for i in range(n):
    for j in range(i + 1, n):
        if heights[i] == heights[j]:
            # Found two buildings with same height
            count = 2
            interval = j - i
            
            # Try to extend the sequence
            pos = j + interval
            while pos < n and heights[pos] == heights[i]:
                count += 1
                pos += interval
            
            max_count = max(max_count, count)

print(max_count)