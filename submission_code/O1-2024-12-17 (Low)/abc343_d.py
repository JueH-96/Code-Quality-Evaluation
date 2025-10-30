def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, T = map(int, input_data[:2])
    updates = input_data[2:]
    
    # scores[player] = current score of that player (player indexed from 0 to N-1)
    scores = [0]*N
    
    # freq[score] = how many players currently have 'score'
    from collections import defaultdict
    freq = defaultdict(int)
    freq[0] = N  # initially all N players have score 0
    
    distinct_count = 1  # there's initially exactly 1 distinct score (0)
    
    idx = 0
    out = []
    for _ in range(T):
        A_i = int(updates[idx]); B_i = int(updates[idx+1])
        idx += 2
        
        # Convert to 0-based index
        player_index = A_i - 1
        
        old_score = scores[player_index]
        new_score = old_score + B_i
        
        # Decrease frequency of old_score
        freq[old_score] -= 1
        if freq[old_score] == 0:
            del freq[old_score]
            distinct_count -= 1
        
        # Increase frequency of new_score
        if new_score not in freq:
            freq[new_score] = 0
            distinct_count += 1
        freq[new_score] += 1
        
        # Update the player's score
        scores[player_index] = new_score
        
        # Now the number of distinct scores is distinct_count
        out.append(str(distinct_count))
    
    print("
".join(out))

# Call main() to execute
if __name__ == "__main__":
    main()