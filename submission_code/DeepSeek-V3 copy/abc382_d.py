# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    
    sequences = []
    
    def backtrack(current_sequence, index):
        if index == N:
            if current_sequence[-1] <= M:
                sequences.append(current_sequence.copy())
            return
        last = current_sequence[-1] if current_sequence else 0
        for a in range(last + 10, M + 1):
            current_sequence.append(a)
            backtrack(current_sequence, index + 1)
            current_sequence.pop()
    
    backtrack([], 0)
    
    # Now, we need to generate all possible sequences where the first element is at least 1
    # and the last element is at most M, and each subsequent element is at least 10 more than the previous.
    
    # To generate all possible sequences, we can use a recursive approach.
    # Start with the first element, and for each possible first element, recursively generate the rest.
    
    # But the above backtrack function already does that.
    
    # Now, we need to sort the sequences lexicographically.
    sequences.sort()
    
    # Now, we need to filter out sequences where the last element is greater than M.
    # But the backtrack function already ensures that the last element is <= M.
    
    # So, we can proceed to print the number of sequences and the sequences themselves.
    
    print(len(sequences))
    for seq in sequences:
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()