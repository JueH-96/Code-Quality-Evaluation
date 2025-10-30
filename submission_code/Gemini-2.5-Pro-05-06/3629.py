class Solution:
  def lengthAfterTransformations(self, s: str, t: int) -> int:
    MOD = 10**9 + 7
    ALPHABET_SIZE = 26

    # dp[j] will store the length derived from character ('a'+j)
    # after the current number of transformations.
    # Initialize for 0 transformations: length is 1 for any character.
    # dp[0] for 'a', dp[1] for 'b', ..., dp[ALPHABET_SIZE-1] for 'z'.
    dp = [1] * ALPHABET_SIZE 

    # Perform t transformations
    for _ in range(t):
        # next_dp will store lengths after one more transformation step
        next_dp = [0] * ALPHABET_SIZE
        
        # Rule for characters 'a' through 'y':
        # char ('a'+j) transforms to ('a'+j+1).
        # New length of ('a'+j) is the length 'dp[j+1]' from the current state.
        for j in range(ALPHABET_SIZE - 1):  # j from 0 ('a') to 24 ('y')
            next_dp[j] = dp[j+1]
        
        # Rule for character 'z':
        # 'z' (index ALPHABET_SIZE-1) transforms to "ab".
        # New length of 'z' is sum of lengths from 'a' (dp[0]) and 'b' (dp[1])
        # from the current state.
        next_dp[ALPHABET_SIZE - 1] = (dp[0] + dp[1]) % MOD
        
        # Update dp array to the new state for the next iteration
        dp = next_dp

    # After t transformations, dp[j] holds L(('a'+j), t).
    # Calculate the total length of the final string by summing up
    # the lengths corresponding to each character in the initial string s.
    total_length = 0
    for char_in_s in s:
        char_code = ord(char_in_s) - ord('a')
        total_length = (total_length + dp[char_code]) % MOD
            
    return total_length