# YOUR CODE HERE
import sys

def main():
    """
    Solves the problem by pre-calculating prefix/suffix match lengths
    and using a frequency-based counting method.
    """
    # Use a faster method for reading input lines
    input = sys.stdin.readline

    try:
        # Read N and T from the first line
        line1 = input().split()
        if not line1:
            return
        N = int(line1[0])
        T = line1[1]
    except (IOError, ValueError, IndexError):
        # Handle cases with empty input or malformed first line
        return
    
    # Read the N strings
    S = [input().strip() for _ in range(N)]

    M = len(T)

    def calc_prefix_len(s: str, t: str) -> int:
        """
        Calculates the length of the longest prefix of t that is a subsequence of s.
        """
        m = len(t)
        t_idx = 0  # Current index in t we are trying to match
        s_idx = 0  # Current index in s
        
        while t_idx < m and s_idx < len(s):
            if s[s_idx] == t[t_idx]:
                t_idx += 1
            s_idx += 1
        return t_idx

    def calc_suffix_len(s: str, t: str) -> int:
        """
        Calculates the length of the longest suffix of t that is a subsequence of s.
        """
        m = len(t)
        matched_len = 0
        s_idx = len(s) - 1  # Current index in s, from the end
        t_idx = m - 1       # Current index in t, from the end
        
        while matched_len < m and s_idx >= 0:
            if s[s_idx] == t[t_idx]:
                matched_len += 1
                t_idx -= 1
            s_idx -= 1
        return matched_len

    # P[i] stores the length of the longest prefix of T that is a subsequence of S[i]
    P = [calc_prefix_len(s, T) for s in S]

    # Q[j] stores the length of the longest suffix of T that is a subsequence of S[j]
    Q = [calc_suffix_len(s, T) for s in S]

    # The condition for a pair (i, j) to be valid is P[i] + Q[j] >= M.
    # We need to count pairs (i, j) satisfying this, which is Q[j] >= M - P[i].
    
    # freq_q[k] stores the number of strings S_j for which Q[j] = k.
    freq_q = [0] * (M + 1)
    for q_val in Q:
        freq_q[q_val] += 1

    # suffix_sum_q[k] stores the number of strings S_j for which Q[j] >= k.
    suffix_sum_q = [0] * (M + 2)
    for k in range(M, -1, -1):
        suffix_sum_q[k] = suffix_sum_q[k + 1] + freq_q[k]

    # For each S_i, add the number of S_j that satisfy the condition.
    total_count = 0
    for p_val in P:
        # We need Q[j] >= M - p_val.
        required_q_len = M - p_val
        
        # Since 0 <= p_val <= M, we have 0 <= required_q_len <= M.
        # The number of j's where Q[j] >= required_q_len is suffix_sum_q[required_q_len].
        if 0 <= required_q_len <= M:
            total_count += suffix_sum_q[required_q_len]

    print(total_count)

if __name__ == "__main__":
    main()