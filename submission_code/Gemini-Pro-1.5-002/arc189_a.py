def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    mod = 998244353
    
    def get_initial_grid(n):
        grid = []
        for i in range(1, n + 1):
            grid.append(i % 2)
        return grid
    
    def apply_operation(grid, l, r):
        new_grid = grid[:]
        for i in range(l + 1, r):
            new_grid[i] = new_grid[l]
        return new_grid

    def is_valid_operation(grid, l, r):
        if l + 1 >= r:
            return False
        if grid[l] != grid[r-1]:
            return False
        for i in range(l + 1, r):
            if grid[i] == grid[l]:
                return False
        return True

    def count_sequences(current_grid, target_grid):
        if current_grid == target_grid:
            return 1
        
        count = 0
        for l in range(n):
            for r in range(l + 2, n + 1):
                if is_valid_operation(current_grid, l, r):
                    new_grid = apply_operation(current_grid, l, r)
                    count = (count + count_sequences(new_grid, target_grid)) % mod
        return count

    initial_grid = get_initial_grid(n)
    result = count_sequences(initial_grid, a)
    print(result)

solve()