class Solution:
  def longestSemiRepetitiveSubstring(self, s: str) -> int:
    n = len(s)
    
    # Constraints state: 1 <= s.length <= 50. So, n is always at least 1.
    # A substring of length 1 is always semi-repetitive (it has 0 pairs of identical consecutive digits).
    # Therefore, max_len will be at least 1 for any valid input s.
    
    # Initialize max_len to 0. The loops will correctly update it.
    # For n=1 (e.g., s="7"):
    # Outer loop: i=0.
    #   Inner loop: j=0.
    #     Substring is s[0:1] ("7"). Length 1.
    #     (j > i) is (0 > 0), which is false. pairs_count remains 0.
    #     pairs_count (0) <= 1 is true.
    #     current_length = 0 - 0 + 1 = 1.
    #     max_len becomes max(0, 1) = 1.
    # Loop finishes. Returns 1. This is correct.
    max_len = 0
    
    # Iterate over all possible start indices `i` of a substring.
    for i in range(n):
      pairs_count = 0
      # Iterate over all possible end indices `j` of a substring s[i...j].
      # The substring s[i...j] corresponds to Python slice s[i:j+1].
      for j in range(i, n):
        # The character s[j] is being "added" to extend the substring from s[i...j-1] to s[i...j].
        # We need to check if this new character s[j] forms a pair with the character s[j-1].
        # This check is only relevant if j > i, which means the substring s[i...j] has length at least 2.
        # If j == i, the substring is s[i:i+1] (e.g., s[0:1]), which has length 1.
        # In this case, (j > i) is false, so pairs_count is not incremented for a single-character substring.
        if j > i and s[j-1] == s[j]:
          pairs_count += 1
        
        # A string is semi-repetitive if it has at most one consecutive pair of same digits.
        if pairs_count <= 1:
          # The current substring s[i...j] is semi-repetitive.
          # Calculate its length and update max_len if this is longer.
          current_length = j - i + 1
          if current_length > max_len:
            max_len = current_length
        else:
          # The current substring s[i...j] has more than one pair (pairs_count > 1).
          # It is not semi-repetitive.
          # Any substring starting at `i` and ending beyond `j` (i.e., s[i...k] where k > j)
          # will also contain these pairs and thus will not be semi-repetitive either.
          # So, we can break from this inner loop (over `j`) and move to the next starting index `i`.
          break
          
    return max_len