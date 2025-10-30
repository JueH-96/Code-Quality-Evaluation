def solve():
    n, m = map(int, input().split())
    conditions = []
    for _ in range(m):
        l, r, x = map(int, input().split())
        conditions.append({'l': l, 'r': r, 'x': x})
    
    valid_permutations = 0
    import itertools
    
    elements = list(range(1, n + 1))
    all_permutations = list(itertools.permutations(elements))
    
    for p_tuple in all_permutations:
        p = list(p_tuple)
        is_valid = True
        for condition in conditions:
            l, r, x = condition['l'], condition['r'], condition['x']
            segment = p[l-1:r]
            max_val = max(segment)
            if max_val == p[x-1]:
                is_valid = False
                break
        if is_valid:
            valid_permutations += 1
            
    print(valid_permutations % 998244353)

if __name__ == '__main__':
    solve()