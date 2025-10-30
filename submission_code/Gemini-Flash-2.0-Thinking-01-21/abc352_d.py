def solve():
    n, k = map(int, input().split())
    p = list(map(int, input().split()))
    if k == 1:
        print(0)
        return
    min_diff = float('inf')
    positions = {}
    for i in range(n):
        positions[p[i]] = i + 1
    
    for start_val in range(1, n - k + 2):
        target_values = [start_val + i for i in range(k)]
        indices = []
        possible = True
        for val in target_values:
            if val not in positions:
                possible = False
                break
            indices.append(positions[val])
        if not possible:
            continue
        indices.sort()
        diff = indices[-1] - indices[0]
        min_diff = min(min_diff, diff)
        
    print(min_diff)

if __name__ == '__main__':
    solve()