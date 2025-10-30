def max_kagamimochi(sizes):
    n = len(sizes)
    used = [False] * n
    count = 0
    
    # Process from largest to smallest (as potential bottoms)
    for i in range(n - 1, -1, -1):
        if used[i]:
            continue
        
        # Find the largest available top that can be placed on this bottom
        for j in range(i - 1, -1, -1):
            if not used[j] and sizes[j] <= sizes[i] / 2:
                used[j] = True
                used[i] = True
                count += 1
                break
    
    return count

# Read input
n = int(input())
sizes = list(map(int, input().split()))
q = int(input())

for _ in range(q):
    l, r = map(int, input().split())
    # Convert to 0-indexed and extract subarray
    l -= 1
    r -= 1
    sub_sizes = sizes[l:r+1]
    # Compute and print answer
    answer = max_kagamimochi(sub_sizes)
    print(answer)