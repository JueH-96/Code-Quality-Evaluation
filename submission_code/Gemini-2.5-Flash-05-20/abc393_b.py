# YOUR CODE HERE
import sys

def solve():
    S = sys.stdin.readline().strip()
    N = len(S)
    
    count = 0
    
    # Iterate through all possible starting positions i_0 (0-indexed)
    # i_0 must be at least 0.
    # The last possible character for 'A' must allow for 'B' and 'C'
    # after it with at least d=1.
    # So, k_0 = i_0 + 2*d. With minimum d=1, k_0 = i_0 + 2.
    # k_0 must be less than N, so i_0 + 2 < N => i_0 < N - 2.
    # Therefore, i_0 ranges from 0 to N-3.
    for i_0 in range(N - 2): # i_0 goes from 0 up to N-3 (inclusive)
        # Check if the character at i_0 is 'A'
        if S[i_0] != 'A':
            continue # If not 'A', this i_0 cannot be the start of a valid triple
            
        # Iterate through all possible common differences d
        # d must be at least 1.
        # The ending index k_0 = i_0 + 2*d must be within the string bounds (i.e., < N).
        # i_0 + 2*d < N
        # 2*d < N - i_0
        # d < (N - i_0) / 2
        # Since d must be an integer, the maximum value for d is (N - i_0 - 1) // 2.
        # The range function's end is exclusive, so we add 1 to include the max value.
        for d in range(1, (N - i_0 - 1) // 2 + 1):
            j_0 = i_0 + d
            k_0 = i_0 + 2 * d
            
            # Check if characters at j_0 and k_0 match 'B' and 'C' respectively
            if S[j_0] == 'B' and S[k_0] == 'C':
                count += 1 # If all conditions met, increment count
                
    print(count)

# Call the solve function to execute the program
solve()