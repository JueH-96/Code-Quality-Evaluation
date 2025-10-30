import collections

class Solution:
  """
  Solves the problem of finding the minimum number of balanced substrings 
  a string can be partitioned into using dynamic programming.
  
  A balanced string is defined as a string where each character that appears 
  in the string occurs the same number of times. For example, "aabb", "abc", "a" 
  are balanced, while "aab", "aabbc" are not.
  """
  def minimumSubstringsInPartition(self, s: str) -> int:
    """
    Calculates the minimum number of balanced substrings that the input string `s` 
    can be partitioned into.

    Args:
        s: The input string, consisting of lowercase English letters. 
           The length of s is guaranteed to be between 1 and 1000.

    Returns:
        The minimum number of balanced substrings required to partition s.
    """
    n = len(s)
    # Initialize DP array. dp[i] will store the minimum number of balanced 
    # substrings needed to partition the prefix s[0...i-1] (first i characters).
    # We use float('inf') to represent states that haven't been reached yet or are impossible.
    dp = [float('inf')] * (n + 1)
    
    # Base case: An empty prefix (length 0) requires 0 substrings.
    dp[0] = 0  

    # Iterate through all possible end positions `i` for prefixes s[:i].
    # `i` represents the length of the prefix, ranging from 1 to n.
    for i in range(1, n + 1):
        # For each prefix ending at index i-1 (length i), we want to find dp[i].
        # To do this, we consider all possible substrings s[j:i] that could be 
        # the last substring in an optimal partition of s[:i].
        
        # Use a collections.Counter to efficiently track character frequencies 
        # within the potential last substring s[j:i].
        counts = collections.Counter() 
        
        # Iterate `j` backwards from `i-1` down to `0`. 
        # `j` represents the starting index of the potential last substring s[j:i].
        # Iterating backwards allows us to incrementally build the frequency count 
        # for s[j:i] as j decreases (i.e., as the substring extends to the left).
        for j in range(i - 1, -1, -1):
            # Add the character s[j] to the frequency counter. 
            # Now `counts` reflects the character frequencies in the substring s[j:i].
            counts[s[j]] += 1
            
            # Check if the current substring s[j:i] is balanced.
            # A substring is balanced if all characters that are present in it 
            # appear exactly the same number of times.
            
            # To check for balance, get the set of unique frequency values 
            # present in the substring s[j:i]. The `counts.values()` method 
            # returns an iterable of the frequencies of characters present.
            unique_freqs = set(counts.values())
            
            # The substring s[j:i] is balanced if and only if there is exactly 
            # one unique frequency count among the characters present.
            # Note: Since j < i, the substring s[j:i] is never empty, so `counts` 
            # will not be empty, and `counts.values()` will contain positive integers.
            is_balanced = (len(unique_freqs) == 1)
            
            # If the substring s[j:i] is balanced:
            if is_balanced:
                # This means we found a valid last piece for a partition of s[:i].
                # A potential partition is formed by taking an optimal partition 
                # of the prefix s[:j] (which requires dp[j] substrings) and 
                # appending the balanced substring s[j:i].
                # The total number of substrings in this partition would be dp[j] + 1.
                
                # We only consider this update if the prefix s[:j] is itself partitionable,
                # which means dp[j] must be a finite number (not infinity).
                if dp[j] != float('inf'):
                    # Update dp[i] to the minimum number of partitions found so far for s[:i].
                    dp[i] = min(dp[i], dp[j] + 1)

    # After iterating through all possible `i` and `j`, the value dp[n] will hold 
    # the minimum number of balanced substrings needed to partition the entire string s (s[:n]).
    # Since any string can be partitioned (e.g., into single characters, which are always balanced),
    # dp[n] is guaranteed to be a finite positive integer.
    return dp[n]