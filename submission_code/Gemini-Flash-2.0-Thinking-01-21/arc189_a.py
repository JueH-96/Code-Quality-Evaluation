def solve():
    n = int(input())
    a = list(map(int, input().split()))
    target_grid = tuple(a)
    initial_grid = tuple([(i % 2) for i in range(1, n + 1)])
    
    memo = {}
    
    def get_valid_operations(grid):
        valid_ops = []
        for l in range(1, n - 1):
            for r in range(l + 2, n + 1):
                if grid[l-1] == grid[r-1]:
                    value = grid[l-1]
                    is_valid_condition = True
                    for i in range(l, r - 1):
                        if grid[i] == value:
                            is_valid_condition = False
                            break
                    if is_valid_condition:
                        valid_ops.append((l, r))
        return valid_ops

    def apply_operation(grid, operation):
        l, r = operation
        value = grid[l-1]
        next_grid_list = list(grid)
        for i in range(l, r - 1):
            next_grid_list[i] = value
        return tuple(next_grid_list)

    def count_sequences_recursive(current_grid):
        if current_grid == target_grid:
            return 1
        if current_grid in memo:
            return memo[current_grid]
        
        count = 0
        valid_operations = get_valid_operations(current_grid)
        for operation in valid_operations:
            next_grid = apply_operation(current_grid, operation)
            count = (count + count_sequences_recursive(next_grid)) % 9982444353
            
        memo[current_grid] = count
        return count

    result = count_sequences_recursive(initial_grid)
    print(result)

if __name__ == '__main__':
    solve()