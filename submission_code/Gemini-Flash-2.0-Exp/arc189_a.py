def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    initial_grid = [(i % 2) for i in range(1, n + 1)]
    
    def apply_operation(grid, l, r):
        new_grid = grid[:]
        for i in range(l + 1, r):
            new_grid[i] = grid[l]
        return new_grid
    
    def is_valid_operation(grid, l, r):
        if l + 1 >= r:
            return False
        if grid[l] != grid[r]:
            return False
        for i in range(l + 1, r):
            if grid[i] == grid[l]:
                return False
        return True
    
    def find_all_sequences(initial_grid, target_grid):
        count = 0
        
        def backtrack(current_grid, sequence):
            nonlocal count
            
            if current_grid == target_grid:
                count = (count + 1) % 998244353
                return
            
            for l in range(n):
                for r in range(l + 2, n):
                    if is_valid_operation(current_grid, l, r):
                        new_grid = apply_operation(current_grid, l, r)
                        backtrack(new_grid, sequence + [(l, r)])
        
        backtrack(initial_grid, [])
        return count

    print(find_all_sequences(initial_grid, a))

solve()