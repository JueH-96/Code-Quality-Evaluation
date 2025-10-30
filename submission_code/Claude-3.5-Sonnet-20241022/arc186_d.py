def is_polish_sequence(seq, dp, n):
    if len(seq) == 0:
        return True
    if len(seq) == 1:
        return seq[0] == 0
    
    key = tuple(seq)
    if key in dp:
        return dp[key]
    
    first = seq[0]
    if first == 0 or first >= n:
        dp[key] = False
        return False
    
    curr_pos = 1
    remaining_seqs = first
    
    while remaining_seqs > 0 and curr_pos < len(seq):
        start_pos = curr_pos
        
        # Try to find a valid Polish sequence
        for end_pos in range(start_pos, len(seq) + 1):
            if is_polish_sequence(seq[start_pos:end_pos], dp, n):
                curr_pos = end_pos
                remaining_seqs -= 1
                break
        
        if curr_pos == start_pos:  # No valid sequence found
            dp[key] = False
            return False
    
    dp[key] = (remaining_seqs == 0 and curr_pos == len(seq))
    return dp[key]

def count_polish_sequences(n, A):
    MOD = 998244353
    
    # dp[i] will store count of valid Polish sequences up to position i
    dp = [0] * (n + 1)
    dp[0] = 1  # Empty sequence
    dp[1] = 1  # Sequence with just 0
    
    # For each length
    for length in range(2, n + 1):
        # For each possible first number
        for first in range(min(length, A[0] + 1)):
            if first == 0:
                continue
                
            # Check if we can form a valid sequence starting with 'first'
            curr_count = dp[first]  # Ways to arrange 'first' number of subsequences
            if curr_count == 0:
                continue
                
            remaining_len = length - 1
            if remaining_len < first:  # Need at least 'first' positions for subsequences
                continue
                
            # Multiply by ways to arrange remaining positions
            curr_count = (curr_count * dp[remaining_len]) % MOD
            dp[length] = (dp[length] + curr_count) % MOD
            
    return dp[n]

# Read input
N = int(input())
A = list(map(int, input().split()))

# Calculate and print result
result = count_polish_sequences(N, A)
print(result)