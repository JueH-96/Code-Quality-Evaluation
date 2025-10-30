import sys

def solve():
    # Read input N
    N = int(sys.stdin.readline())
    # Read input string S
    S = sys.stdin.readline().strip()
    
    # Convert string S to a list of integers (0 or 1)
    # A[0] corresponds to A_1, A[N-1] corresponds to A_N in the problem statement (1-based indexing)
    A = [int(c) for c in S]

    # Initialize total sum
    total_sum = 0
    
    # State variables:
    # count0_prev: number of starting indices i in [1, j-1] such that f(i, j-1) = 0
    # count1_prev: number of starting indices i in [1, j-1] such that f(i, j-1) = 1
    # These counts summarize the results f(i, j-1) for the previous ending index j-1.
    # Initialize for j=1 (previous step j-1=0), where there are no i in [1, 0].
    count0_prev = 0
    count1_prev = 0

    # Iterate through the ending index j from 1 to N
    # For each j, we calculate sum(f(i, j) for i=1 to j) and update counts for the next step.
    for j in range(1, N + 1):
        # Get the value of A_j (j-th element in 1-based indexing)
        Aj = A[j-1]

        # Calculate the sum of f(i, j) for the current j
        Sj = 0
        
        # Calculate the counts for the current j: f(i, j) for i=1 to j
        count0_curr = 0 # count of i in [1, j] such that f(i, j)=0
        count1_curr = 0 # count of i in [1, j] such that f(i, j)=1

        if Aj == 0:
            # If A_j is 0:
            # For any starting index i in [1, j-1], f(i, j) = f(i, j-1) NAND 0 = 1.
            # There are j-1 such terms, all are 1.
            Sj += (j - 1) # Add sum of f(i, j) for 1 <= i <= j-1
            
            # For the starting index i=j, f(j, j) = A_j = 0.
            Sj += 0       # Add f(j, j)
            
            # Update counts for the current j:
            # f(i, j) = 1 for 1 <= i <= j-1 (j-1 terms)
            count1_curr = (j - 1)
            # f(j, j) = 0 (1 term)
            count0_curr = 1

        else: # Aj == 1
            # If A_j is 1:
            # For any starting index i in [1, j-1], f(i, j) = f(i, j-1) NAND 1 = NOT f(i, j-1).
            # The values f(1,j), ..., f(j-1,j) are the flipped values of f(1,j-1), ..., f(j-1,j-1).
            # The number of 1s among f(1,j)...f(j-1,j) is the number of 0s among f(1,j-1)...f(j-1,j-1), which was count0_prev.
            Sj += count0_prev # Add sum of f(i, j) for 1 <= i <= j-1
            
            # For the starting index i=j, f(j, j) = A_j = 1.
            Sj += 1           # Add f(j, j)

            # Update counts for the current j:
            # For 1 <= i <= j-1: f(i, j) = 0 if f(i, j-1) = 1. Number of such i is count1_prev.
            count0_curr = count1_prev
            # For 1 <= i <= j-1: f(i, j) = 1 if f(i, j-1) = 0. Number of such i is count0_prev.
            # Add the case f(j, j) = 1.
            count1_curr = count0_prev + 1

        # Add the sum calculated for the current j to the total sum across all j
        total_sum += Sj

        # The counts calculated for the current j (f(i, j) for i=1..j) will be the 'prev' counts for the next iteration j+1.
        count0_prev = count0_curr
        count1_prev = count1_curr

    # Print the final total sum
    print(total_sum)

solve()