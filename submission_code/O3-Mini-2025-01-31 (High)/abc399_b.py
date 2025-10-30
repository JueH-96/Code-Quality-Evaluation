def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    scores = list(map(int, data[1:]))

    # Create list of tuples (score, original_index)
    persons = [(score, i) for i, score in enumerate(scores)]
    # Sort in descending order of score
    persons.sort(key=lambda x: x[0], reverse=True)
    
    # Prepare an array to hold the rank results.
    result = [0] * N
    
    current_rank = 1
    i = 0
    while i < N:
        current_score = persons[i][0]
        # Count persons with the same score.
        j = i
        while j < N and persons[j][0] == current_score:
            j += 1
        count = j - i  # number of persons with the current highest score
        
        # Assign current_rank to all persons in this group.
        for k in range(i, j):
            # persons[k][1] is the original index
            result[persons[k][1]] = current_rank
        
        # Increase current_rank by count (as described in the problem)
        current_rank += count
        # Move to the next group of scores.
        i = j

    # Print the results line by line.
    sys.stdout.write("
".join(map(str, result)) + "
")
    
if __name__ == '__main__':
    main()