# YOUR CODE HERE
import sys

def solve():
    """
    This function encapsulates the entire logic for solving the problem.
    It reads from stdin, processes the queries, and writes to stdout.
    """
    
    # Reading input from standard input
    s = sys.stdin.readline().strip()
    q_count = int(sys.stdin.readline())
    queries = list(map(int, sys.stdin.readline().split()))
    
    s_len = len(s)
    
    results = []
    
    # Process each query
    for k_val in queries:
        # The problem uses 1-based indexing for K_i, so we convert to 0-based.
        k = k_val - 1
        
        # The core of the problem lies in understanding the structure of the string S
        # after a huge number of operations. The operation is:
        # S_{i+1} = S_i + swapcase(S_i)
        
        # Let S_0 be the initial string. T_0 = swapcase(S_0).
        # S_1 = S_0 + T_0
        # S_2 = S_1 + swapcase(S_1) = (S_0 + T_0) + swapcase(S_0 + T_0)
        #     = (S_0 + T_0) + (T_0 + S_0)
        # S_3 = S_2 + swapcase(S_2) = (S_0 T_0 T_0 S_0) + (T_0 S_0 S_0 T_0)
        
        # If we look at the string in blocks of the original length `s_len`,
        # let B_n be the n-th block (0-indexed).
        # The sequence of blocks is S_0, T_0, T_0, S_0, T_0, S_0, S_0, T_0, ...
        # This corresponds to the Thue-Morse sequence.
        # The n-th term of the Thue-Morse sequence is 0 if the number of 1s in the
        # binary representation of n (popcount) is even, and 1 if it's odd.
        
        # Let's map S_0 to 0 and T_0 to 1.
        # B_n = S_0 if popcount(n) is even.
        # B_n = T_0 = swapcase(S_0) if popcount(n) is odd.
        
        # A character at a 0-indexed position `k` is located in block `n = k // s_len`
        # at the relative position `r = k % s_len`.
        # The base character is `s[r]`.
        # The number of swapcase operations applied to this character is determined
        # by the parity of popcount(n). If popcount(n) is odd, one effective swap
        # is performed. If even, zero effective swaps.
        
        # Therefore, we can find the character for any k by:
        # 1. Calculating its block number `n` and remainder `r`.
        # 2. Calculating the popcount of `n`.
        # 3. Applying swapcase to `s[r]` if the popcount is odd.
        
        # The number of operations 10^100 is large enough that for any K_i up to 10^18,
        # the character is determined by this infinite Thue-Morse structure.
        
        # A special case might be k=0. Here, n=0, r=0. popcount(0)=0 (even).
        # So the character is s[0]. This is correct, as the first character of S
        # is never changed by the operation S := S + T.
        
        # For any k < s_len, n=0, so popcount(0)=0. The character is s[k], which is also correct.
        
        # The calculation for a given k:
        n, r = divmod(k, s_len)
        
        # Python's `bin(n).count('1')` is a convenient way to get the popcount.
        # For performance with very large numbers, this is efficient.
        swaps = bin(n).count('1')
        
        char = s[r]
        if swaps % 2 == 1:
            char = char.swapcase()
            
        results.append(char)
        
    # Print all results in a single line, separated by spaces.
    print(*results)

# Enclosing the call to the main function in a standard Python construct.
if __name__ == "__main__":
    solve()