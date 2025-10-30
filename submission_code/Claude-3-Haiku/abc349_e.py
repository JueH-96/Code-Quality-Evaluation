from typing import List

def read_input() -> List[List[int]]:
    grid = []
    for _ in range(3):
        row = list(map(int, input().split()))
        grid.append(row)
    return grid

def check_win(grid: List[List[int]], color: str) -> bool:
    # Check rows
    for row in grid:
        if all(cell == color for cell in row):
            return True
    # Check columns
    for j in range(3):
        if all(grid[i][j] == color for i in range(3)):
            return True
    # Check diagonals
    if all(grid[i][i] == color for i in range(3)):
        return True
    if all(grid[i][2-i] == color for i in range(3)):
        return True
    return False

def play_game(grid: List[List[int]]) -> str:
    takahashi_score = 0
    aoki_score = 0
    takahashi_turn = True
    while True:
        # Find the next available white cell
        for i in range(3):
            for j in range(3):
                if grid[i][j] == 0:
                    break
            else:
                continue
            break
        else:
            # No more white cells, compare scores
            if takahashi_score > aoki_score:
                return "Takahashi"
            else:
                return "Aoki"

        # Play the move
        score = grid[i][j]
        if takahashi_turn:
            grid[i][j] = "R"
            takahashi_score += score
        else:
            grid[i][j] = "B"
            aoki_score += score

        # Check for a win
        if check_win(grid, "R"):
            return "Takahashi"
        if check_win(grid, "B"):
            return "Aoki"

        takahashi_turn = not takahashi_turn

def main():
    grid = read_input()
    print(play_game(grid))

if __name__ == "__main__":
    main()