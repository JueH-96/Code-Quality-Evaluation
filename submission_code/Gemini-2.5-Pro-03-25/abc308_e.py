# YOUR CODE HERE
import sys

def mex(a, b, c):
    """
    Computes the minimum excluded value (mex) for a set of three integers {a, b, c}.
    The mex is the smallest non-negative integer not present in the set.
    Since a, b, c are from {0, 1, 2}, the mex can be 0, 1, 2, or 3.
    """
    seen = {a, b, c}
    # Check for 0, 1, 2 in order to find the minimum missing non-negative integer
    if 0 not in seen:
        return 0
    if 1 not in seen:
        return 1
    if 2 not in seen:
        return 2
    # If 0, 1, and 2 are all present in the set {a, b, c}
    return 3

def solve():
    """
    Reads input, computes the required sum based on the problem specification, 
    and prints the result to standard output.
    """
    N = int(sys.stdin.readline())
    # Read the sequence A, converting elements to integers
    A = list(map(int, sys.stdin.readline().split()))
    # Read the string S
    S = sys.stdin.readline().strip()

    # Calculate prefix sums for 'M' characters combined with values from A.
    # PrefixM[idx][val] stores the count of indices p < idx 
    # such that S[p] == 'M' and A[p] == val.
    # The table size is (N+1) x 3. PrefixM[0] is base case [0,0,0].
    # Index `idx` corresponds to the state *after* considering elements up to index `idx-1`.
    PrefixM = [[0] * 3 for _ in range(N + 1)]
    for idx in range(N):
        # Copy counts from the previous index state PrefixM[idx]
        for val in range(3):
            PrefixM[idx+1][val] = PrefixM[idx][val]
        
        # If the character at index idx is 'M', increment the count for its corresponding value A[idx]
        if S[idx] == 'M':
            # A[idx] is guaranteed to be 0, 1, or 2 based on constraints
            PrefixM[idx+1][A[idx]] += 1

    # Calculate suffix sums for 'X' characters combined with values from A.
    # SuffixX[idx][val] stores the count of indices p >= idx
    # such that S[p] == 'X' and A[p] == val.
    # The table size is (N+1) x 3. SuffixX[N] is base case [0,0,0].
    # Index `idx` corresponds to the state considering elements from index `idx` to `N-1`.
    SuffixX = [[0] * 3 for _ in range(N + 1)]
    # Iterate backwards from N-1 down to 0
    for idx in range(N - 1, -1, -1):
        # Copy counts from the next index state SuffixX[idx+1]
        for val in range(3):
            SuffixX[idx][val] = SuffixX[idx+1][val]
        
        # If the character at index idx is 'X', increment the count for its corresponding value A[idx]
        if S[idx] == 'X':
            # A[idx] is guaranteed to be 0, 1, or 2 based on constraints
            SuffixX[idx][A[idx]] += 1

    # Calculate the total sum required by the problem
    total_sum = 0
    # Iterate through each index j from 0 to N-1 to consider it as the middle element ('E') of a 'MEX' triple
    for j in range(N):
        # Check if the character at index j is 'E'
        if S[j] == 'E':
            # Get the value A[j] for the middle element
            a_j = A[j] 
            
            # Iterate through all possible values p (0, 1, 2) for A[i] where i < j
            for p in range(3): 
                # Get the count of indices i < j such that S[i] == 'M' and A[i] == p.
                # This count is available from the precomputed prefix sums: PrefixM[j][p].
                countM = PrefixM[j][p] 
                
                # If there are no 'M' indices with value p before j, we cannot form a valid triple
                # involving this p value. Continue to the next possible value for p.
                if countM == 0:
                    continue
                
                # Iterate through all possible values q (0, 1, 2) for A[k] where k > j
                for q in range(3): 
                    # We need the count of indices k > j such that S[k] == 'X' and A[k] == q.
                    # This condition is equivalent to finding count of indices k >= j+1 such that S[k] == 'X' and A[k] == q.
                    # This count is available from the precomputed suffix sums: SuffixX[j+1][q].
                    
                    # SuffixX table has N+1 elements, indexed 0 to N.
                    # The maximum value for j+1 is N (when j=N-1). So SuffixX[j+1] is always a valid index lookup.
                    countX = SuffixX[j+1][q] 
                    
                    # If there are no 'X' indices with value q after j, we cannot form a valid triple
                    # involving this q value. Continue to the next possible value for q.
                    if countX == 0:
                        continue

                    # Calculate the mex value for the triplet of values (p, a_j, q).
                    # These values correspond potentially to (A[i], A[j], A[k]) for some i < j < k.
                    current_mex = mex(p, a_j, q)
                    
                    # Add the contribution to the total sum.
                    # For a fixed j with S[j]='E' and A[j]=a_j:
                    # Each distinct pair (i, k) such that i < j, S[i]='M', A[i]=p AND k > j, S[k]='X', A[k]=q
                    # forms a valid triple (i, j, k) satisfying the conditions S_i S_j S_k = 'MEX'.
                    # The number of such pairs (i, k) is countM * countX.
                    # Each such triple contributes `current_mex` to the total sum.
                    total_sum += countM * countX * current_mex

    # Print the final computed total sum
    print(total_sum)

# Execute the main solution function when the script is run
solve()