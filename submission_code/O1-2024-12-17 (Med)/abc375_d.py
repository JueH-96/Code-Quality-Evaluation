def main():
    import sys
    S = sys.stdin.readline().strip()
    n = len(S)
    
    # For a 3-letter string to be a palindrome (xyz), we need x == z.
    # We want to count the number of triples (i, j, k) with i < j < k and S[i] == S[k].
    # Approach:
    #   1. Collect the (1-based) positions of each character.
    #   2. For each character's positions sorted in ascending order:
    #         let positions = [p1, p2, ..., pm].
    #         We'll sum for each k from 2..m:
    #            contribution = Σ_{i=1..(k-1)} [ (p_k - 1) - p_i ] 
    #         This counts how many choices of (i, j) there are for each position p_k,
    #         with i in [1..(k-1)]. The j can be any integer between (p_i+1) .. (p_k - 1).
    #         Number of ways = (p_k - 1 - p_i).
    #
    #      We can do this quickly using prefix sums:
    #         prefix_sum[i] = p1 + p2 + ... + pi.
    #         Then for position p_k, the sum of [p_k - 1 - p_i] over i=1..(k-1) is:
    #            (p_k - 1)*(k-1) - prefix_sum[k-1]
    
    # Create an array of lists to store positions of each character A..Z
    letter_positions = [[] for _ in range(26)]
    for idx, ch in enumerate(S):
        letter_positions[ord(ch) - ord('A')].append(idx + 1)
    
    answer = 0
    
    # Calculate the contribution for each character's positions
    for cpos in letter_positions:
        # If fewer than 2 positions for this character, it can't form x y x with different i,k
        if len(cpos) < 2:
            continue
        
        # Build prefix sums of positions
        prefix_sum = [0] * (len(cpos) + 1)
        for i in range(len(cpos)):
            prefix_sum[i+1] = prefix_sum[i] + cpos[i]
        
        # For each k from 2..len(cpos), the contribution is:
        #   (cpos[k-1] - 1) * (k-1) - prefix_sum[k-1]
        for k in range(2, len(cpos) + 1):
            answer += (cpos[k-1] - 1) * (k - 1) - prefix_sum[k-1]
    
    print(answer)

# Do not forget to call main()
main()