def solve():
    n, m = map(int, input().split())
    conditions = []
    for _ in range(m):
        l, r, x = map(int, input().split())
        conditions.append(((l, r), x))
    
    total_permutations = 0
    
    def is_valid_permutation(p):
        for condition_range, x_index in conditions:
            l, r = condition_range
            max_val = -1
            for i in range(l - 1, r):
                max_val = max(max_val, p[i])
            if max_val == p[x_index - 1]:
                return False
        return True
        
    import itertools
    
    elements = list(range(1, n + 1))
    valid_count = 0
    
    for p_tuple in itertools.permutations(elements):
        p_list = list(p_tuple)
        if is_valid_permutation(p_list):
            valid_count += 1
            
    print(valid_count % 998244353)

if __name__ == '__main__':
    solve()