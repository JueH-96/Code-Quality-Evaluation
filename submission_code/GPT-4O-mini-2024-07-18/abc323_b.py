# YOUR CODE HERE
def rank_players():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    results = data[1:N+1]
    
    wins = []
    
    for i in range(N):
        win_count = results[i].count('o')
        wins.append((win_count, i + 1))  # Store (win_count, player_number)
    
    # Sort by number of wins (descending) and player number (ascending)
    wins.sort(key=lambda x: (-x[0], x[1]))
    
    # Extract the sorted player numbers
    sorted_players = [player for _, player in wins]
    
    print(" ".join(map(str, sorted_players)))

rank_players()