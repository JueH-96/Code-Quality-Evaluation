# YOUR CODE HERE
import sys

def solve():
    # Read input string S from standard input
    S = sys.stdin.readline().strip()
    N = len(S)
    
    # dp_prev_idx stores the 1-based index in the target pattern P='ioio...' where the *previous* character of S was matched.
    # The target pattern P has 'i' at odd indices (1, 3, 5, ...) and 'o' at even indices (2, 4, 6, ...).
    # Initialize dp_prev_idx to 0. This represents the state before matching any character from S. A value of 0 means the next character can potentially start matching from index 1.
    dp_prev_idx = 0 
    
    # Iterate through each character of the input string S using 0-based index k
    for k in range(N):
        current_char = S[k]
        
        # Calculate the tentative next available index in the pattern string P.
        # This is the smallest index strictly greater than the index where the previous character was matched.
        # Since indices are 1-based, the smallest index > dp_prev_idx is dp_prev_idx + 1.
        j_tentative = dp_prev_idx + 1
        
        # This variable will store the actual 1-based index where the current character S[k] is matched in P.
        current_dp_idx = 0 
        
        # Determine the required properties of the index j where S[k] should be matched.
        if current_char == 'i':
            # The character 'i' must match a position with an odd index in P (positions 1, 3, 5, ...).
            # We need the smallest odd index j such that j > dp_prev_idx.
            
            # Check if the tentatively chosen index j_tentative meets the requirement (is odd).
            # The parity of an index j is determined by j % 2. Odd indices have j % 2 == 1.
            if j_tentative % 2 == 1: 
                # If j_tentative is odd, it satisfies the requirement for 'i' and is the smallest such index > dp_prev_idx. Use it.
                current_dp_idx = j_tentative
            else: 
                # If j_tentative is even, it's not a valid position for 'i'. The next index (j_tentative + 1) must be odd.
                # This j_tentative + 1 will be the smallest odd index > dp_prev_idx because j_tentative was the smallest index > dp_prev_idx overall. Use j_tentative + 1.
                current_dp_idx = j_tentative + 1
        else: # current_char == 'o'
            # The character 'o' must match a position with an even index in P (positions 2, 4, 6, ...).
            # We need the smallest even index j such that j > dp_prev_idx.

            # Check if the tentatively chosen index j_tentative meets the requirement (is even).
            # Even indices have j % 2 == 0.
            if j_tentative % 2 == 0: 
                # If j_tentative is even, it satisfies the requirement for 'o' and is the smallest such index > dp_prev_idx. Use it.
                current_dp_idx = j_tentative
            else: 
                # If j_tentative is odd, it's not a valid position for 'o'. The next index (j_tentative + 1) must be even.
                # This j_tentative + 1 will be the smallest even index > dp_prev_idx. Use it.
                current_dp_idx = j_tentative + 1
        
        # Update dp_prev_idx to the index where the current character S[k] was matched.
        # This value will be used in the next iteration to find the appropriate position for S[k+1], ensuring indices are increasing.
        dp_prev_idx = current_dp_idx

    # After processing all characters in S, dp_prev_idx holds the 1-based index of the position in P 
    # where the last character of S (S[N-1]) was matched. Let's call this final_idx.
    # This final_idx value represents the minimum length M' of a prefix of P (i.e., P[1...M']) 
    # such that S is its subsequence, with each character of S placed at an index in P that matches the required pattern 
    # ('i' at odd positions, 'o' at even positions).
    final_idx = dp_prev_idx
    
    # The problem requires the final constructed string T to have an even length M.
    # Additionally, T must contain S as a subsequence according to the pattern rules. This implies that T must be 
    # a prefix of P and its length M must be at least final_idx to accommodate the match of S found.
    # Therefore, we need to find the smallest even integer M such that M >= final_idx.
    M = 0
    if final_idx % 2 == 0: 
        # If final_idx is already even, it is the smallest possible length M satisfying both conditions (M >= final_idx and M is even).
        M = final_idx
    else: 
        # If final_idx is odd, the smallest even integer greater than or equal to final_idx is final_idx + 1.
        M = final_idx + 1
        
    # The minimum number of insertions required is the difference between the required length M of the final valid string T
    # and the original length N of the input string S. The characters in T that are not part of the subsequence S
    # correspond exactly to the characters that must be inserted.
    insertions = M - N
    
    # Print the final calculated minimum number of insertions to standard output.
    print(insertions)

# Call the solve function to execute the program logic using the standard input.
solve()