def solve():
    n, k = map(int, input().split())
    functions = []
    for _ in range(n):
        functions.append(list(map(int, input().split())))
    
    max_result = -float('inf')
    
    import itertools
    
    indices = list(range(n))
    permutations = itertools.permutations(indices, k)
    
    for p_indices in permutations:
        current_value = 1
        for i in range(k - 1, -1, -1):
            func_index = p_indices[i]
            a_i, b_i = functions[func_index]
            current_value = a_i * current_value + b_i
        max_result = max(max_result, current_value)
        
    print(max_result)

if __name__ == '__main__':
    solve()