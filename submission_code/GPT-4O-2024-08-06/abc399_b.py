# YOUR CODE HERE
def determine_ranks(N, scores):
    # Create a list of tuples (score, index) to keep track of original indices
    indexed_scores = [(scores[i], i) for i in range(N)]
    
    # Sort the list based on scores in descending order
    indexed_scores.sort(reverse=True, key=lambda x: x[0])
    
    # Prepare a list to store ranks
    ranks = [0] * N
    
    # Initialize rank variable
    r = 1
    
    # Iterate over the sorted scores
    i = 0
    while i < N:
        # Get the current score
        current_score = indexed_scores[i][0]
        
        # Count how many people have this score
        count = 0
        while i + count < N and indexed_scores[i + count][0] == current_score:
            count += 1
        
        # Assign the current rank to all people with this score
        for j in range(count):
            original_index = indexed_scores[i + j][1]
            ranks[original_index] = r
        
        # Update the rank for the next group
        r += count
        
        # Move to the next group of scores
        i += count
    
    return ranks

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    N = int(data[0])
    scores = list(map(int, data[1:]))
    
    result = determine_ranks(N, scores)
    
    for rank in result:
        print(rank)