def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    initial_grid = [(i % 2) for i in range(1, n + 1)]
    
    if initial_grid == a:
        print(1)
        return

    
    def check_valid_operation(grid, l, r):
        if l + 1 >= r:
            return False
        if grid[l] != grid[r]:
            return False
        for i in range(l + 1, r):
            if grid[i] == grid[l]:
                return False
        return True

    def apply_operation(grid, l, r):
        new_grid = list(grid)
        for i in range(l + 1, r):
            new_grid[i] = new_grid[l]
        return new_grid
    
    def find_all_sequences(current_grid, target_grid, current_sequence, all_sequences):
        if current_grid == target_grid:
            all_sequences.append(list(current_sequence))
            return
        
        for l in range(n):
            for r in range(l + 1, n):
                if check_valid_operation(current_grid, l, r):
                    new_grid = apply_operation(current_grid, l, r)
                    current_sequence.append((l, r))
                    find_all_sequences(new_grid, target_grid, current_sequence, all_sequences)
                    current_sequence.pop()
    
    all_sequences = []
    find_all_sequences(initial_grid, a, [], all_sequences)
    
    print(len(all_sequences) % 998244353)

solve()