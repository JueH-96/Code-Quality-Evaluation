import copy

def solve():
    h, w, x = map(int, input().split())
    p, q = map(int, input().split())
    initial_strengths = []
    for _ in range(h):
        initial_strengths.append(list(map(int, input().split())))
    
    memo = {}
    
    def get_max_strength(current_grid, current_pos):
        grid_tuple = tuple(tuple(row) for row in current_grid)
        state = (grid_tuple, current_pos)
        if state in memo:
            return memo[state]
        
        current_p, current_q = current_pos
        current_strength = current_grid[current_p-1][current_q-1]
        possible_absorptions = []
        
        adjacent_positions = [(current_p-1, current_q), (current_p+1, current_q), (current_p, current_q-1), (current_p, current_q+1)]
        for next_p, next_q in adjacent_positions:
            if 1 <= next_p <= h and 1 <= next_q <= w:
                adjacent_strength = current_grid[next_p-1][next_q-1]
                if adjacent_strength > 0 and adjacent_strength < (current_strength / x):
                    possible_absorptions.append((next_p, next_q))
                    
        if not possible_absorptions:
            memo[state] = current_strength
            return current_strength
            
        max_strength_achieved = current_strength
        for next_p, next_q in possible_absorptions:
            next_grid = copy.deepcopy(current_grid)
            absorbed_strength = next_grid[next_p-1][next_q-1]
            next_grid[next_p-1][next_q-1] = current_strength + absorbed_strength
            next_grid[current_p-1][current_q-1] = 0
            
            result_strength = get_max_strength(next_grid, (next_p, next_q))
            max_strength_achieved = max(max_strength_achieved, result_strength)
            
        memo[state] = max_strength_achieved
        return max_strength_achieved
        
    result = get_max_strength(initial_strengths, (p, q))
    print(result)

if __name__ == '__main__':
    solve()