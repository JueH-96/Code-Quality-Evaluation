def count_buildings(n, heights):
    counts = []
    
    for i in range(n):
        count = 0
        for j in range(i+1, n):
            valid = True
            for k in range(i+1, j):
                if heights[k] > heights[j]:
                    valid = False
                    break
            if valid:
                count += 1
        counts.append(count)
    
    return counts

# Read input
n = int(input())
heights = list(map(int, input().split()))

# Compute counts
counts = count_buildings(n, heights)

# Print result
print(" ".join(map(str, counts)))