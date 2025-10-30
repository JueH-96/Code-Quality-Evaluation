def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    M = int(input_data[1])
    votes = list(map(int, input_data[2:2 + M]))
    
    # Array to store the count of votes for each candidate
    counts = [0] * (N + 1)
    current_winner = 0  # will hold the candidate with the highest votes so far
    best_count = 0      # maximum vote count seen so far

    result = []
    
    for vote in votes:
        counts[vote] += 1
        # Check if this vote makes the candidate have more votes than the previous best.
        if counts[vote] > best_count:
            best_count = counts[vote]
            current_winner = vote
        # If it's a tie, the candidate with the lower candidate number wins.
        elif counts[vote] == best_count and vote < current_winner:
            current_winner = vote
        result.append(str(current_winner))
    
    sys.stdout.write("
".join(result))

if __name__ == '__main__':
    main()