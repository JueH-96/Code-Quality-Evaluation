# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    # Define the winning moves
    win = {'R': 'P', 'P': 'S', 'S': 'R'}
    
    # Initialize Takahashi's moves and the count of wins
    takahashi = []
    wins = 0
    
    # Iterate through each game
    for i in range(N):
        aoki_move = S[i]
        # Determine the possible moves for Takahashi
        # He must not lose, so he can choose the winning move or the same move
        # But to maximize wins, we choose the winning move
        # Also, he must alternate his moves
        if i == 0:
            # First move, choose the winning move
            takahashi_move = win[aoki_move]
        else:
            # Subsequent moves, must be different from the previous move
            # Choose the winning move if possible, otherwise choose the same move
            # But since he must not lose, he can choose the winning move or the same move
            # To maximize wins, choose the winning move if it's different from the previous move
            if win[aoki_move] != takahashi[-1]:
                takahashi_move = win[aoki_move]
            else:
                # If the winning move is the same as the previous move, choose the same move as Aoki
                takahashi_move = aoki_move
        takahashi.append(takahashi_move)
        if takahashi_move == win[aoki_move]:
            wins += 1
    
    print(wins)

if __name__ == "__main__":
    main()