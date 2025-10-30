def max_wins(n, s):
    """
    Determine the maximum number of games Takahashi could have won.
    
    Parameters:
    n (int): The number of games.
    s (str): Aoki's moves represented by a string of length N consisting of the characters R, P, and S.
    
    Returns:
    int: The maximum number of games Takahashi could have won.
    """
    
    # Initialize the count of wins
    wins = 0
    
    # Initialize the previous move of Takahashi
    prev_move = None
    
    # Iterate over Aoki's moves
    for move in s:
        # Determine the winning move for Takahashi
        if move == 'R':
            winning_move = 'P'
        elif move == 'P':
            winning_move = 'S'
        else:
            winning_move = 'R'
        
        # If the winning move is different from the previous move, increment the count of wins
        if winning_move != prev_move:
            wins += 1
            prev_move = winning_move
        # If the winning move is the same as the previous move, try the losing move
        else:
            if winning_move == 'R':
                losing_move = 'S'
            elif winning_move == 'P':
                losing_move = 'R'
            else:
                losing_move = 'P'
            # If the losing move is different from the previous move, increment the count of wins
            if losing_move != prev_move:
                wins += 1
                prev_move = losing_move
    
    return wins

# Read the input from stdin
n = int(input())
s = input()

# Print the maximum number of games Takahashi could have won
print(max_wins(n, s))