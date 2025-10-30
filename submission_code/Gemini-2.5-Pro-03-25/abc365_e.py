import sys

# Define the main function to encapsulate the solution logic
def solve():
    # Read the integer N from standard input, which is the length of the sequence.
    N = int(sys.stdin.readline())
    # Read the sequence A of N integers from standard input.
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate the prefix XOR sums of the sequence A.
    # P is an array of size N+1. P[0] is initialized to 0.
    # P[i] stores the XOR sum A[0] ^ A[1] ^ ... ^ A[i-1] using 0-based indexing for A.
    # In terms of 1-based indexing used in the problem statement (A_1, ..., A_N),
    # P[k] corresponds to A_1 ^ A_2 ^ ... ^ A_k for k > 0, and P[0] = 0.
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = P[i] ^ A[i]

    # Initialize the total sum, which will store the final answer.
    total_sum = 0
    
    # Iterate through each bit position k from 0 up to 59.
    # The maximum value of A_i is 10^8, which is less than 2^27. 
    # Therefore, considering bits up to k=26 would be sufficient.
    # Using k up to 59 covers standard 64-bit integer ranges and is safe.
    for k in range(60): 
        # Initialize counts for the k-th bit of prefix XOR sums P[m].
        # C0: count of indices m (0 <= m <= N) where the k-th bit of P[m] is 0.
        # C1: count of indices m (0 <= m <= N) where the k-th bit is 1.
        C0 = 0
        C1 = 0
        
        # Initialize count Tk for the k-th bit of original array elements A[m].
        # Tk represents the number of elements A_m (1 <= m <= N) whose k-th bit is 1.
        # In 0-based indexing for A, this is the count of A[m] (0 <= m <= N-1) where the k-th bit is 1.
        Tk = 0
        
        # Iterate through the prefix XOR sums P (indices 0 to N).
        # Compute C0 and C1 for the current bit position k.
        for m in range(N + 1):
            # Check the k-th bit of P[m]. 
            # (P[m] >> k) shifts the k-th bit to the least significant position.
            # & 1 extracts this bit (0 or 1).
            if (P[m] >> k) & 1:
                C1 += 1 # If the k-th bit is 1, increment C1
            else:
                C0 += 1 # If the k-th bit is 0, increment C0
            
        # Iterate through the original array A (indices 0 to N-1).
        # Compute Tk for the current bit position k.
        for m in range(N):
             # Check the k-th bit of A[m].
            if (A[m] >> k) & 1:
                Tk += 1 # If the k-th bit is 1, increment Tk
        
        # Calculate Sk based on the derived formula S_k = C_0 * C_1 - T_k.
        # S_k represents the count of pairs (i, j) with 1 <= i < j <= N such that the k-th bit of the XOR sum (A_i ^ ... ^ A_j) is 1.
        # The derivation relies on relating the sum over pairs (i, j) with i < j to the sum over pairs (p, q) with p < q,
        # using the property X(i, j) = P[i-1] ^ P[j].
        # The total count of pairs (p, q) with 0 <= p < q <= N where the k-th bits of P[p] and P[q] differ is C0 * C1.
        # This count includes pairs where q = p+1. The count of such pairs where bits differ is Tk.
        # Sk is the count of pairs where q >= p+2, thus Sk = (Total pairs p<q with different bits) - (Pairs p, p+1 with different bits) = C0*C1 - Tk.
        Sk = C0 * C1 - Tk
        
        # Add the contribution of the k-th bit position to the total sum.
        # The contribution is Sk multiplied by the value represented by the k-th bit, which is 2^k.
        # (1 << k) computes 2^k efficiently.
        total_sum += Sk * (1 << k)

    # Print the final computed total sum to standard output.
    print(total_sum)

# Call the solve function to execute the main logic of the program.
solve()