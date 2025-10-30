def find_winner(n, s):
    """
    Find the overall winner between Takahashi and Aoki.

    Args:
    n (int): The number of games played.
    s (str): A string representing the results of the games.

    Returns:
    str: The overall winner, either 'T' or 'A'.
    """
    takahashi_wins = 0
    aoki_wins = 0

    # Iterate over the string to count the wins for each player
    for char in s:
        if char == 'T':
            takahashi_wins += 1
        else:
            aoki_wins += 1

    # If Takahashi won more games, he is the overall winner
    if takahashi_wins > aoki_wins:
        return 'T'
    # If Aoki won more games, he is the overall winner
    elif aoki_wins > takahashi_wins:
        return 'A'
    # If they won the same number of games, the overall winner is the one who reached that number of wins first
    else:
        # Iterate over the string again to find the first player to reach the number of wins
        takahashi_count = 0
        aoki_count = 0
        for char in s:
            if char == 'T':
                takahashi_count += 1
            else:
                aoki_count += 1
            # If Takahashi reached the number of wins first, he is the overall winner
            if takahashi_count > aoki_count:
                return 'T'
            # If Aoki reached the number of wins first, he is the overall winner
            elif aoki_count > takahashi_count:
                return 'A'

# Read the input from stdin
n = int(input())
s = input()

# Find and print the overall winner
print(find_winner(n, s))