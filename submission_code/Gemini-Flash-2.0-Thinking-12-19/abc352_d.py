def solve():
    n, k = map(int, input().split())
    p = list(map(int, input().split()))
    if k == 1:
        print(0)
        return
    min_diff = float('inf')
    for start_val in range(1, n - k + 2):
        target_values = set(range(start_val, start_val + k))
        indices = []
        value_index_map = {}
        for i in range(n):
            value_index_map[p[i]] = i + 1
        for val in target_values:
            if val in value_index_map:
                indices.append(value_index_map[val])
        if len(indices) < k:
            continue
        indices.sort()
        for i in range(len(indices) - k + 1):
            current_indices = indices[i:i+k]
            diff = current_indices[-1] - current_indices[0]
            min_diff = min(min_diff, diff)
            
    print(min_diff)

if __name__ == '__main__':
    solve()