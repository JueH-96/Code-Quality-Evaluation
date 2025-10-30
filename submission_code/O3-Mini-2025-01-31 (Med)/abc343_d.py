def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    T = int(next(it))
    
    # Initialize scores for all N players (0-indexed).
    scores = [0] * N
    # Maintain a frequency dictionary for score values.
    freq = {0: N}
    
    results = []
    for _ in range(T):
        a = int(next(it))
        b = int(next(it))
        idx = a - 1  # Convert 1-indexed to 0-indexed.
        
        # Get old score and update with new points.
        old_score = scores[idx]
        new_score = old_score + b
        scores[idx] = new_score
        
        # Update frequency of the old score.
        freq[old_score] -= 1
        if freq[old_score] == 0:
            del freq[old_score]
        
        # Update frequency of the new score.
        if new_score in freq:
            freq[new_score] += 1
        else:
            freq[new_score] = 1
        
        # Append the count of different score values.
        results.append(str(len(freq)))
    
    # Output the results line by line.
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()