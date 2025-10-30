def main():
    import sys
    from functools import lru_cache

    a_flat = []
    for _ in range(3):
        a_flat += list(map(int, sys.stdin.readline().split()))
    
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]

    def check_win(grid):
        for line in lines:
            a, b, c = line
            if grid[a] == grid[b] == grid[c]:
                if grid[a] == 1:
                    return 'Takahashi'
                elif grid[a] == 2:
                    return 'Aoki'
        return None

    @lru_cache(maxsize=None)
    def solve(grid):
        grid_list = list(grid)
        winner = check_win(grid_list)
        if winner is not None:
            return winner
        if all(cell != 0 for cell in grid_list):
            taka = sum(a_flat[i] for i, cell in enumerate(grid_list) if cell == 1)
            aoki = sum(a_flat[i] for i, cell in enumerate(grid_list) if cell == 2)
            return 'Takahashi' if taka > aoki else 'Aoki'
        count = sum(1 for cell in grid_list if cell != 0)
        is_takahashi_turn = (count % 2 == 0)
        possible_moves = [i for i in range(9) if grid_list[i] == 0]
        for move in possible_moves:
            new_grid = list(grid_list)
            new_grid[move] = 1 if is_takahashi_turn else 2
            new_grid_tuple = tuple(new_grid)
            result = solve(new_grid_tuple)
            if (is_takahashi_turn and result == 'Takahashi') or (not is_takahashi_turn and result == 'Aoki'):
                return 'Takahashi' if is_takahashi_turn else 'Aoki'
        return 'Aoki' if is_takahashi_turn else 'Takahashi'
    
    initial_state = tuple([0] * 9)
    print(solve(initial_state))

if __name__ == "__main__":
    main()