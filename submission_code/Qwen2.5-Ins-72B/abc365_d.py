# YOUR CODE HERE
def solve(n, s):
    # Mapping of moves to their winning move
    win_map = {'R': 'P', 'P': 'S', 'S': 'R'}
    
    # Initialize the previous move and the count of wins
    prev_move = ''
    wins = 0
    
    # Track the number of consecutive wins that can be made
    consecutive_wins = 0
    
    for move in s:
        # Determine the move that wins against Aoki's move
        winning_move = win_map[move]
        
        # If the winning move is different from the previous move, use it
        if winning_move != prev_move:
            prev_move = winning_move
            wins += 1
            consecutive_wins = 1
        else:
            # If the winning move is the same as the previous move, use a different move
            for alt_move in 'RPS':
                if alt_move != prev_move:
                    prev_move = alt_move
                    break
            consecutive_wins = 0
    
    return wins

# Read input
n = int(input())
s = input()

# Solve and print the result
print(solve(n, s))