N, L, R = map(int, input().split())

def query(i, j):
    print(f"? {i} {j}")
    result = int(input())
    if result == -1:
        exit()
    return result

def get_range_sum(left, right):
    if left == right:
        # Find smallest i where 2^i > left
        i = 0
        while (1 << i) <= left:
            i += 1
        j = left >> i
        return query(i, j)
        
    # Find largest power of 2 that fits in range
    range_len = right - left + 1
    i = 0
    while (1 << (i + 1)) <= range_len:
        i += 1
        
    # Query the largest power of 2 range
    j = left >> i
    curr_sum = query(i, j)
    
    # Recursively get sum of remaining range
    next_left = left + (1 << i)
    if next_left <= right:
        remaining = get_range_sum(next_left, right)
        curr_sum = (curr_sum + remaining) % 100
        
    return curr_sum

result = get_range_sum(L, R)
print(f"! {result}")