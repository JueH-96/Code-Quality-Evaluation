def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:]))
    
    # We need to determine the rank of each person
    ranks = [0] * N
    
    # Create a list of (score, index) tuples
    score_index_pairs = [(P[i], i) for i in range(N)]
    
    # Sort by score descending and by index ascending
    score_index_pairs.sort(reverse=True, key=lambda x: x[0])
    
    rank = 1
    i = 0
    while i < N:
        current_score = score_index_pairs[i][0]
        start = i
        
        # Find all people with the same score
        while i < N and score_index_pairs[i][0] == current_score:
            i += 1
        
        # All people from index `start` to `i-1` have the same score
        for j in range(start, i):
            person_index = score_index_pairs[j][1]
            ranks[person_index] = rank
        
        # Update rank to the next available rank
        rank += (i - start)
    
    # Output the ranks
    for rank in ranks:
        print(rank)

if __name__ == "__main__":
    main()