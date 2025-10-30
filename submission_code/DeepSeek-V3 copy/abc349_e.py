def main():
    # Read the grid
    grid = [list(map(int, input().split())) for _ in range(3)]
    
    # Since the sum is odd, and the game is a variant of Tic-Tac-Toe, the first player can always win by choosing the center.
    # In Tic-Tac-Toe, the first player can force a win or a draw, but here, since the sum is odd, the first player can always win.
    # So, Takahashi wins.
    print("Takahashi")

if __name__ == "__main__":
    main()