import sys

# It is recommended to use fast I/O in Python for competitive programming
readline = sys.stdin.readline

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        N = int(readline())
        A = list(map(int, readline().split()))
    except (IOError, ValueError):
        # This part handles potential empty lines or format errors in input,
        # which can occur in some online judge systems.
        return

    total_ans = 0
    
    # The maximum value of an element is 10^8 (< 2^27). The total sum can be much larger.
    # The sum can be up to ~ (N^2/2) * max_XOR_val.
    # log2((2e5)^2/2 * 2^27) is approx. 61.2, so we should check bits up to 61.
    for d in range(62):
        power_of_2 = 1 << d
        
        # C_d: Count of subarrays of length >= 2 where the d-th bit of XOR sum is 1.
        # C_d = S_d_all - S_d_len1
        
        # S_d_len1: Count for subarrays of length 1 (i.e., single elements).
        s_d_len1 = 0
        for x in A:
            if (x >> d) & 1:
                s_d_len1 += 1
        
        # S_d_all: Count for all subarrays (length >= 1).
        s_d_all = 0
        
        # p_j = d-th bit of (A[0] ^ ... ^ A[j])
        # We maintain counts of 0s and 1s in the sequence of prefix XOR bits seen so far.
        # {p_{-1}, p_{0}, ..., p_{j-1}}
        
        # Start with prefix XOR of an empty sequence, which is 0.
        # So, for p_{-1}, the count of 0s is 1.
        count0 = 1 
        count1 = 0
        
        current_prefix_xor_bit = 0
        for j in range(N):
            # Get the d-th bit of the current element A[j].
            bit_A_j = (A[j] >> d) & 1
            # Update the prefix XOR bit up to A[j].
            current_prefix_xor_bit ^= bit_A_j
            
            # For the current prefix XOR bit p_j, we count how many previous
            # prefix XOR bits p_{i-1} (for i <= j) differ from it.
            if current_prefix_xor_bit == 0:
                s_d_all += count1
            else: # current_prefix_xor_bit == 1
                s_d_all += count0
                
            # Update counts to include the current prefix XOR bit for subsequent steps.
            if current_prefix_xor_bit == 0:
                count0 += 1
            else:
                count1 += 1
        
        count_for_bit_d = s_d_all - s_d_len1
        
        total_ans += power_of_2 * count_for_bit_d
        
    print(total_ans)

solve()