def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, T = map(int, input_data[:2])
    
    # We'll store the current scores of all N players in a list.
    # Initially, everyone has 0 points.
    scores = [0] * N
    
    # We'll use a dictionary (freq) to keep track of how many players
    # have a certain score. Initially, all N players have a score of 0.
    from collections import defaultdict
    freq = defaultdict(int)
    freq[0] = N
    
    # We'll parse the input pairs (A_i, B_i) from the remaining data in input_data.
    idx = 2
    out = []
    for _ in range(T):
        A_i = int(input_data[idx]) - 1  # 0-based index of the player
        B_i = int(input_data[idx + 1])
        idx += 2
        
        # The player's old score
        old_score = scores[A_i]
        # The player's new score
        new_score = old_score + B_i
        scores[A_i] = new_score
        
        # Update freq mapping
        freq[old_score] -= 1
        if freq[old_score] == 0:
            del freq[old_score]
        
        freq[new_score] += 1
        
        # The number of distinct scores is just the number of keys in freq
        out.append(str(len(freq)))
    
    # Print results
    print("
".join(out))

# Don't forget to call main()
if __name__ == "__main__":
    main()