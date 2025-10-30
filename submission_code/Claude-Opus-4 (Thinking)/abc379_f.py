# YOUR CODE HERE
N, Q = map(int, input().split())
heights = [0] + list(map(int, input().split()))  # 1-indexed

for _ in range(Q):
    l, r = map(int, input().split())
    
    # Find buildings > r that are visible from l
    visible_from_l = set()
    max_height = 0
    for j in range(l + 1, N + 1):
        if heights[j] > max_height:
            if j > r:
                visible_from_l.add(j)
            max_height = heights[j]
    
    # Count buildings > r visible from r that are also visible from l
    count = 0
    max_height = 0
    for j in range(r + 1, N + 1):
        if heights[j] > max_height:
            if j in visible_from_l:
                count += 1
            max_height = heights[j]
    
    print(count)