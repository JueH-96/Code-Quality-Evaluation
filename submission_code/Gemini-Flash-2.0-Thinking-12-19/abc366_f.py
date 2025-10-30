def solve():
    n, k = map(int, input().split())
    functions = []
    for _ in range(n):
        functions.append(list(map(int, input().split())))
    
    max_result = -float('inf')
    
    def calculate_value(permutation_indices):
        current_value = 1
        for index in reversed(permutation_indices):
            func_index = index - 1
            a_i, b_i = functions[func_index]
            current_value = a_i * current_value + b_i
        return current_value
        
    def get_permutations(current_permutation, available_indices):
        nonlocal max_result
        if len(current_permutation) == k:
            value = calculate_value(current_permutation)
            max_result = max(max_result, value)
            return
            
        for index in list(available_indices):
            next_permutation = current_permutation + [index]
            next_available_indices = set(available_indices)
            next_available_indices.remove(index)
            get_permutations(next_permutation, next_available_indices)
            
    initial_available_indices = set(range(1, n + 1))
    get_permutations([], initial_available_indices)
    print(max_result)

if __name__ == '__main__':
    solve()