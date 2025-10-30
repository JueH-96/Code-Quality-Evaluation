def solve():
    n = int(input())
    a = list(map(int, input().split()))
    initial_grid = [(i % 2) for i in range(1, n + 1)]
    target_grid = tuple(a)
    
    memo = {}
    
    def get_valid_operations(current_grid_tuple):
        valid_ops = []
        current_grid = list(current_grid_tuple)
        for l in range(n):
            for r in range(l + 2, n):
                if current_grid[l] == current_grid[r]:
                    condition_met = True
                    for i in range(l + 1, r):
                        if current_grid[i] == current_grid[l]:
                            condition_met = False
                            break
                    if condition_met:
                        valid_ops.append((l, r))
        return valid_ops

    def apply_operation(current_grid_tuple, l, r):
        current_grid = list(current_grid_tuple)
        value_to_set = current_grid[l]
        for i in range(l + 1, r):
            current_grid[i] = value_to_set
        return tuple(current_grid)

    def count_sequences(current_grid_tuple):
        if current_grid_tuple == target_grid:
            return 1
        if current_grid_tuple in memo:
            return memo[current_grid_tuple]
        
        valid_operations = get_valid_operations(current_grid_tuple)
        if not valid_operations:
            return 0
            
        count = 0
        for l, r in valid_operations:
            next_grid_tuple = apply_operation(current_grid_tuple, l, r)
            count = (count + count_sequences(next_grid_tuple)) % 998244353
            
        memo[current_grid_tuple] = count
        return count

    result = count_sequences(tuple(initial_grid))
    print(result)

if __name__ == '__main__':
    solve()