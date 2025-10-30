def main():
    import sys
    from collections import defaultdict
    
    data = sys.stdin.read().strip().split()
    N, T = map(int, data[:2])
    events = data[2:]
    
    # freq will keep track of how many players currently have a given score
    freq = defaultdict(int)
    freq[0] = N  # initially, all N players have 0 score
    
    # scores[i] will hold the current score of player i
    scores = [0]*N
    
    idx = 0
    output_lines = []
    for _ in range(T):
        A_i = int(events[idx])
        B_i = int(events[idx + 1])
        idx += 2
        
        player_idx = A_i - 1   # convert to 0-based index
        old_score = scores[player_idx]
        new_score = old_score + B_i
        scores[player_idx] = new_score
        
        # Decrease the count of the old score
        freq[old_score] -= 1
        if freq[old_score] == 0:
            del freq[old_score]
        
        # Increase the count of the new score
        freq[new_score] += 1
        
        # The number of distinct scores is just the number of keys in freq
        output_lines.append(str(len(freq)))
    
    # Print results
    print("
".join(output_lines))

# Call main to ensure the solution runs
if __name__ == "__main__":
    main()