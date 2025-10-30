def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # input_data now holds all tokens

    # Parse first two integers N, Q
    N, Q = map(int, input_data[:2])
    # Next token is the string S
    S = input_data[2]
    
    # Precompute "A": A[i] = 1 if S[i-1] == S[i] (1-based for i), else 0
    # We'll implement this using 0-based indexing for the string.
    # A will be stored as A[i] for i in [1..N-1].
    A = [0]*(N+1)
    for i in range(1, N):
        if S[i-1] == S[i]:
            A[i] = 1
    
    # Build the prefix sum array P where P[i] = A[1] + A[2] + ... + A[i]
    P = [0]*(N+1)
    for i in range(1, N+1):
        P[i] = P[i-1] + A[i]
    
    # Now process each query
    # Each query is of the form (l, r)
    # Answer = P[r-1] - P[l-1]   (for 1-based l,r)
    # We'll extract them from input_data after the first 1+ (N) token for S
    # So queries start at index 3 in input_data
    answers = []
    idx = 3
    for _ in range(Q):
        l = int(input_data[idx]); r = int(input_data[idx+1])
        idx += 2
        
        # Number of places with same letter twice in a row:
        # sum(A[l..r-1]) = P[r-1] - P[l-1]
        if r == l:
            answers.append(0)
        else:
            answers.append(P[r-1] - P[l-1])
    
    # Print the answers
    print("
".join(map(str, answers)))


# Do not forget to call main()
if __name__ == "__main__":
    main()