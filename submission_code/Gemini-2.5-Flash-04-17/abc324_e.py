import sys
import bisect

# Function to compute the length of the longest prefix of pattern (T)
# that is a subsequence of text (S)
# This runs in O(|text|) time.
def longest_prefix_subsequence_length(text, pattern):
    p_text = 0
    p_pattern = 0
    len_text = len(text)
    len_pattern = len(pattern)
    while p_text < len_text and p_pattern < len_pattern:
        if text[p_text] == pattern[p_pattern]:
            p_pattern += 1
        p_text += 1
    return p_pattern # This is the length of the matched prefix

def main():
    # Read N and T from the first line
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    T = line1[1]
    
    # Read N strings S_i
    S = [sys.stdin.readline().strip() for _ in range(N)]
    
    len_T = len(T)
    
    # Compute L_i for each S_i: length of longest prefix of T subsequence of S_i
    L_values = []
    for s in S:
        L_values.append(longest_prefix_subsequence_length(s, T))
        
    # Compute L'_i for each S_i: length of longest prefix of T_rev subsequence of S_i_rev
    # Note: This is equivalent to the length of the longest suffix of T
    # that is a subsequence of S_i.
    T_rev = T[::-1]
    L_prime_values = []
    for s in S:
        # Reversing string `s` inside the loop is fine as total length is constrained
        s_rev = s[::-1]
        L_prime_values.append(longest_prefix_subsequence_length(s_rev, T_rev))
        
    # Count pairs (i, j) such that L_values[i] + L_prime_values[j] >= len_T
    
    # Sort L_prime_values to efficiently count
    L_prime_values_sorted = sorted(L_prime_values)
    
    count = 0
    for L_i in L_values:
        # We need L_prime_values[j] >= len_T - L_i
        required_L_prime = len_T - L_i
        
        # Use bisect_left to find the index of the first element >= required_L_prime
        # in the sorted L_prime_values_sorted list.
        # All elements from this index onwards satisfy the condition.
        idx = bisect.bisect_left(L_prime_values_sorted, required_L_prime)
        
        # The number of elements >= required_L_prime is len(L_prime_values_sorted) - idx
        count += (N - idx)
    
    print(count)

main()