import sys

def read_input():
    """Reads the input from stdin."""
    grid = []
    for _ in range(3):
        row = list(map(int, sys.stdin.readline().split()))
        grid.append(row)
    return grid

def calculate_score(grid):
    """Calculates the total score of the grid."""
    total_score = 0
    for row in grid:
        total_score += sum(row)
    return total_score

def determine_winner(grid):
    """Determines the winner based on the total score."""
    total_score = calculate_score(grid)
    if total_score > 0:
        return "Takahashi"
    else:
        return "Aoki"

def main():
    """Main function."""
    grid = read_input()
    winner = determine_winner(grid)
    print(winner)

if __name__ == "__main__":
    main()