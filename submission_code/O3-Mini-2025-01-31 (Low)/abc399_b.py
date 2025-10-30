def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    # First token is N
    n = int(input_data[0])
    # Next n tokens are scores, convert them to integers
    scores = list(map(int, input_data[1:n+1]))
    
    # Initialize ranks list with zeros for undetermined status
    ranks = [0] * n
    current_rank = 1
    
    # Process until every person's rank is determined (rank > 0)
    while any(rank == 0 for rank in ranks):
        # Find maximum among scores of undetermined persons
        max_score = max(scores[i] for i in range(n) if ranks[i] == 0)
        
        # For each person with that maximum score, assign the current rank
        count = 0
        for i in range(n):
            if ranks[i] == 0 and scores[i] == max_score:
                ranks[i] = current_rank
                count += 1
        
        # Update the current rank by the number of people just ranked
        current_rank += count

    # Output the result
    for rank in ranks:
        sys.stdout.write(f"{rank}
")

if __name__ == '__main__':
    main()