import collections

class Solution:
  def validSequence(self, word1: str, word2: str) -> list[int]:
    n = len(word1)
    k = len(word2)
    inf = float('inf')

    # Precompute next_occurrence table for O(1) lookups.
    # next_occurrence[i][char_code] = smallest index >= i of char
    # Time and Space: O(n * 26)
    next_occurrence = [[inf] * 26 for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for char_code in range(26):
            next_occurrence[i][char_code] = next_occurrence[i + 1][char_code]
        next_occurrence[i][ord(word1[i]) - ord('a')] = i

    def find_next(char_to_find: str, start_idx: int) -> int:
        if start_idx + 1 > n:
            return inf
        return next_occurrence[start_idx + 1][ord(char_to_find) - ord('a')]

    def find_next_mismatch(char_to_avoid: str, start_idx: int) -> int:
        min_idx = inf
        if start_idx + 1 > n:
            return inf
        avoid_code = ord(char_to_avoid) - ord('a')
        for char_code in range(26):
            if char_code != avoid_code:
                min_idx = min(min_idx, next_occurrence[start_idx + 1][char_code])
        return min_idx

    # DP state:
    # dp[j][d]: last index used for word2[:j+1] with d differences.
    # parent_s1[j]: 0 or 1, indicates if s1 at step j came from s0 or s1 at j-1.
    # s0_smaller[j]: True if s0 is lexicographically smaller than s1 at step j.
    dp = [[inf] * 2 for _ in range(k)]
    parent_s1 = [0] * k
    s0_smaller = [False] * k

    # Base case j=0
    dp[0][0] = find_next(word2[0], -1)
    dp[0][1] = find_next_mismatch(word2[0], -1)
    
    s0_valid_base = dp[0][0] != inf
    s1_valid_base = dp[0][1] != inf
    if s0_valid_base and s1_valid_base:
        s0_smaller[0] = dp[0][0] < dp[0][1]
    elif s0_valid_base:
        s0_smaller[0] = True
    
    # Fill DP table
    for j in range(1, k):
        char_j = word2[j]

        # Update s0
        if dp[j - 1][0] != inf:
            dp[j][0] = find_next(char_j, dp[j - 1][0])

        # Update s1
        next_A = inf
        if dp[j - 1][1] != inf:
            next_A = find_next(char_j, dp[j - 1][1])
        valid_A = next_A != inf

        next_B = inf
        if dp[j - 1][0] != inf:
            next_B = find_next_mismatch(char_j, dp[j - 1][0])
        valid_B = next_B != inf
        
        use_A = False
        if valid_A and not valid_B:
            use_A = True
        elif not valid_A and valid_B:
            use_A = False
        elif valid_A and valid_B:
            use_A = not s0_smaller[j-1]
        else: # both invalid
            dp[j][1] = inf
            s0_smaller[j] = dp[j][0] != inf
            continue

        if use_A:
            dp[j][1] = next_A
            parent_s1[j] = 1
        else: # use B
            dp[j][1] = next_B
            parent_s1[j] = 0

        # Update s0_smaller[j]
        s0_valid = dp[j][0] != inf
        s1_valid = dp[j][1] != inf

        if s0_valid and s1_valid:
            if use_A:
                s0_smaller[j] = False
            else: # use B
                s0_smaller[j] = dp[j][0] < dp[j][1]
        elif s0_valid:
            s0_smaller[j] = True
        # else: s0_smaller[j] remains False

    # Reconstruct the result
    final_s0_valid = dp[k - 1][0] != inf
    final_s1_valid = dp[k - 1][1] != inf

    if not final_s0_valid and not final_s1_valid:
        return []

    end_d = 0
    if final_s0_valid and final_s1_valid:
        end_d = 0 if s0_smaller[k - 1] else 1
    elif final_s0_valid:
        end_d = 0
    else: # final_s1_valid
        end_d = 1
        
    res = [0] * k
    curr_d = end_d
    for j in range(k - 1, -1, -1):
        res[j] = dp[j][curr_d]
        if j > 0:
            if curr_d == 1:
                curr_d = parent_s1[j]
    
    return res