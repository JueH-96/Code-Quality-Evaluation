def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    # Count occurrences of each value
    count = {}
    for val in a:
        count[val] = count.get(val, 0) + 1
    
    result = 0
    
    # Count pairs where a_i = a_j
    for val, cnt in count.items():
        if cnt >= 2:
            result += cnt * (cnt - 1) // 2
    
    # Count special pairs (1, 2)
    if 1 in count and 2 in count:
        result += count[1] * count[2]
    
    return result

t = int(input())
for _ in range(t):
    print(solve())