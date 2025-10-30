def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    # Parse input
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))
    
    # Initialize vote counts. Use 1-indexed list.
    counts = [0] * (N + 1)
    
    # Variables to keep track of the current winner candidate and its vote count.
    # Start with candidate = None and max_votes = 0.
    current_winner = None
    current_max = 0
    
    # Prepare a list to store results; later output them joined by newline.
    results = []
    
    for _ in range(M):
        vote = int(next(it))
        counts[vote] += 1
        # If this vote makes the candidate equal or exceed the current maximum votes.
        if counts[vote] > current_max:
            # New maximum found, update winner & current_max.
            current_max = counts[vote]
            current_winner = vote
        elif counts[vote] == current_max:
            # If there's a tie, choose the candidate with smallest candidate number.
            # Only update if vote candidate is less than current_winner.
            # Note: if current_winner is None, then update (but initially it'll be assigned).
            if current_winner is None or vote < current_winner:
                current_winner = vote
        
        # Append the current winner to the result.
        results.append(str(current_winner))
    
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()