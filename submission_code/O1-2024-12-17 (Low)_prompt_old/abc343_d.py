def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    N, T = map(int, input_data[:2])
    updates = input_data[2:]
    
    from collections import defaultdict
    
    # Dictionary to count how many players have a given score
    freq = defaultdict(int)
    # Initially, all N players have 0 points
    freq[0] = N
    
    # Array to store current scores of each player
    # (Player i's score is stored at index i-1)
    scores = [0] * N
    
    idx = 0
    pos = 0
    
    # We will collect answers and then print them at the end
    results = []
    
    for _ in range(T):
        A_i = int(updates[pos])
        B_i = int(updates[pos + 1])
        pos += 2
        
        player_index = A_i - 1
        old_score = scores[player_index]
        new_score = old_score + B_i
        
        # Update the player's score
        scores[player_index] = new_score
        
        # Decrease the frequency of the old score
        freq[old_score] -= 1
        if freq[old_score] == 0:
            del freq[old_score]
        
        # Increase the frequency of the new score
        freq[new_score] += 1
        
        # The number of distinct scores is the number of keys in freq
        results.append(str(len(freq)))
    
    # Print all results, one per line
    print("
".join(results))

# Call solve()
solve()