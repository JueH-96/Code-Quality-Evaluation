# YOUR CODE HERE

def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    # Parse the first line
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # The next token is the initial string S
    S = list(input_data[2])
    
    # Precompute how many times "ABC" appears in S initially
    count_abc = 0
    for i in range(N - 2):
        if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
            count_abc += 1
    
    # A helper function to check if "ABC" occurs at index i
    def is_abc(i):
        if i < 0 or i > N - 3:
            return False
        return (S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C')
    
    # We'll read queries from input_data as well
    index = 3  # we've already consumed 3 tokens (N, Q, and S)
    answers = []
    
    for _ in range(Q):
        x = int(input_data[index]); index += 1
        c = input_data[index];      index += 1
        
        p = x - 1  # convert to 0-based index
        
        # Remove occurrences of "ABC" that overlap with position p
        for j in (p-2, p-1, p):
            if is_abc(j):
                count_abc -= 1
        
        # Update the character in S
        S[p] = c
        
        # Add back occurrences of "ABC" that exist after the update
        for j in (p-2, p-1, p):
            if is_abc(j):
                count_abc += 1
        
        # Record the answer for this query
        answers.append(str(count_abc))
    
    # Print the answers for all queries
    print("
".join(answers))

# Do not forget to call main()
if __name__ == "__main__":
    main()