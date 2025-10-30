# YOUR CODE HERE
import sys

def solve():
    # Read N and T from the first line of input
    # N_str is used because N could be large, though within standard integer limits usually.
    # Reading as string first then converting avoids potential issues, but int() should be fine.
    N_str, T = sys.stdin.readline().split()
    N = int(N_str)
    M = len(T)
    
    # Read N strings S_1, ..., S_N using strip() to remove potential trailing newline characters
    S = [sys.stdin.readline().strip() for _ in range(N)]

    # P[i] will store the length of the longest prefix of T that is a subsequence of S[i]
    P = [0] * N 
    # L[i] will store the length of the longest suffix of T that is a subsequence of S[i]
    L = [0] * N 

    # Compute P values for all strings S[i]
    for i in range(N):
        k = 0 # k represents the length of the prefix of T matched so far, T[0...k-1]
        current_s = S[i]
        len_s = len(current_s)
        
        # Iterate through the characters of S[i]
        for s_idx in range(len_s):
             # If we haven't matched the entire T yet (k < M) and 
             # the current character in S[i] matches the next required character in T (T[k])
             if k < M and current_s[s_idx] == T[k]:
                 k += 1 # Increment the length of the matched prefix
        # After iterating through S[i], k holds the length of the longest prefix found
        P[i] = k


    # Compute L values for all strings S[i]
    for i in range(N):
        k = 0 # k represents the length of the suffix of T matched so far, T[M-k...M-1]
        current_s = S[i]
        len_s = len(current_s)
        
        # Iterate through the characters of S[i] in reverse order
        for s_idx in range(len_s - 1, -1, -1):
            # If we haven't matched the entire T suffix yet (k < M) and
            # the current character in S[i] matches the next required character from the end of T (T[M-1-k])
            # T[M-1-k] is the character at index M-1-k, which is the (k+1)-th character from the end.
            # Example: If k=0, we are looking for T[M-1] (last char). If k=1, T[M-2] (second last char), etc.
            if k < M and current_s[s_idx] == T[M-1-k]:
                 k += 1 # Increment the length of the matched suffix
        # After iterating through S[i] backwards, k holds the length of the longest suffix found
        L[i] = k
        

    # Compute frequency counts of the L values.
    # countL[k] will store the number of indices j such that L[j] = k.
    # The possible values for L[j] are from 0 to M.
    countL = [0] * (M + 1)
    for j in range(N):
        # L[j] is the computed length for S[j], which is guaranteed to be between 0 and M.
        countL[L[j]] += 1

    # Compute suffix sums of the frequency counts.
    # suffixSumL[k] will store the total count of indices j such that L[j] >= k.
    # This is calculated efficiently using dynamic programming / prefix sums in reverse.
    # We use an array of size M+2 to handle the boundary case k=M easily. suffixSumL[M+1] is implicitly 0.
    suffixSumL = [0] * (M + 2) 
    for k in range(M, -1, -1):
        # The number of strings with suffix length >= k is the sum of those with length k
        # and those with length >= k+1.
        suffixSumL[k] = suffixSumL[k+1] + countL[k]

    # Calculate the total number of pairs (i, j) that satisfy the condition.
    # The condition is that the concatenation S[i] + S[j] contains T as a subsequence.
    # This condition is equivalent to P[i] + L[j] >= M.
    total_pairs = 0
    for i in range(N):
        # For a fixed i, we need to find the number of j such that P[i] + L[j] >= M.
        # Rearranging gives: L[j] >= M - P[i].
        
        required_min_L = M - P[i]
        
        # Since P[i] is the length of the longest prefix match, 0 <= P[i] <= M.
        # Therefore, 0 <= M - P[i] <= M. So, 0 <= required_min_L <= M.
        
        # We need to count the number of indices j such that L[j] >= required_min_L.
        # This count is already precomputed in suffixSumL[required_min_L].
        
        # Add this count to the total number of pairs.
        # The index required_min_L is guaranteed to be valid for suffixSumL array (indices 0 to M).
        total_pairs += suffixSumL[required_min_L]

    # Print the final computed total number of pairs.
    print(total_pairs)

# Execute the main solve function to run the program.
solve()