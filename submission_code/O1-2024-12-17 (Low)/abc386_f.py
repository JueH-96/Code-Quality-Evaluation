def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    K = int(input_data[0])
    S = input_data[1]
    T = input_data[2]

    # Quick check: if |len(S) - len(T)| > K, it's impossible 
    # to make them the same in <= K edits.
    n, m = len(S), len(T)
    if abs(n - m) > K:
        print("No")
        return

    # Banded edit distance (Levenshtein) approach in O(K * (n+m)).
    # We'll keep a rolling DP array "old_dp" of size m+1, where
    # old_dp[j] = edit distance between S[:i] and T[:j], for the
    # current i. Then we move to the next i and construct "new_dp".
    
    # old_dp[j] for i=0 (S is empty):
    # cost is j insertions to get T[:j].
    old_dp = list(range(m+1))
    
    # Large value to mark "out of band" positions:
    INF = 10**9
    
    for i in range(1, n+1):
        # new_dp will store dp for S[:i], T[:j]
        new_dp = [INF]*(m+1)
        new_dp[0] = i  # cost of removing i chars to transform S[:i] -> ""
        
        # We only need to compute j in the band [i-K..i+K]
        j_low = max(1, i - K)
        j_high = min(m, i + K)
        
        s_char = S[i-1]  # current character in S
        
        # Within that band, compute dp:
        # new_dp[j] = min of:
        #   old_dp[j] + 1        (delete from S)
        #   new_dp[j-1] + 1      (insert into S or skip in T)
        #   old_dp[j-1] + cost   (replace if needed)
        # We have to handle indices carefully.
        
        prev_val = INF  # Will hold new_dp[j-1] as we move forward
        for j in range(j_low, j_high+1):
            # cost of substitution:
            cost = 0 if s_char == T[j-1] else 1
            
            # insertion (T's j-th char is inserted to match S[i]):
            ins = new_dp[j-1] + 1 if j > 0 else INF
            
            # deletion (remove s_char from S):
            dele = old_dp[j] + 1
            
            # substitution / match:
            sub = old_dp[j-1] + cost if j > 0 else INF
            
            # choose minimal
            val = ins
            if dele < val:
                val = dele
            if sub < val:
                val = sub
            new_dp[j] = val
        
        old_dp = new_dp
        
        # Minor early exit if distance already <= K at the far end:
        # (not strictly necessary, but might be slightly faster in some cases)
        if old_dp[m] <= K:
            # We still finish the loop to finalize the dp, but
            # in practice we can break if i == n anyway.
            pass

    # After processing all n characters, 
    # old_dp[m] = edit distance between S[:n], T[:m].
    if old_dp[m] <= K:
        print("Yes")
    else:
        print("No")

# Call main() at the end
main()