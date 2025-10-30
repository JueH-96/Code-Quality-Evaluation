class Solution:
  def smallestString(self, s: str) -> str:
    n = len(s)
    chars = list(s)

    # Find the starting index 'i' of the segment to modify.
    # This is the first character that is not 'a'.
    # All characters s[0...i-1] (if any) are 'a's and will be untouched.
    i = 0
    while i < n and chars[i] == 'a':
        i += 1

    # Case 1: The string consists of all 'a's (e.g., "aaa", "a").
    # 'i' will be equal to 'n' in this case.
    if i == n:
        # We must perform the operation exactly once.
        # To make the string lexicographically smallest, we should make the change
        # as far to the right as possible. If we change s[k]='a' to 'z',
        # picking k=n-1 results in "aa...az", which is the smallest possible.
        chars[n-1] = 'z'
        return "".join(chars)

    # Case 2: The string has at least one non-'a' character.
    # 'i' is the index of the first non-'a' character.
    # We will start our operation at this index.
    # s[0...i-1] are all 'a's and remain untouched.
    # s[i] is not 'a'. We change it to s[i]-1. This makes it smaller.
    
    # Now, we determine the end of the segment to modify.
    # We iterate from index 'i' (inclusive) onwards. Let this iterator be 'j'.
    # We continue decrementing characters s[j] as long as s[j] is not 'a'.
    # If s[j] is not 'a', changing it to s[j]-1 makes this position smaller (good).
    # If s[j] is 'a', changing it to 'z' makes this position larger (bad), so we stop.
    # The segment s[i...j-1] will consist of characters that were originally non-'a'.
    
    j = i
    # Note: chars[i] is guaranteed to be non-'a' at this point because of the previous loop
    # and the i == n check. So the following loop will execute at least once (for index i).
    while j < n and chars[j] != 'a':
        chars[j] = chr(ord(chars[j]) - 1) # Modify character in place
        j += 1
    
    # All characters s[k] for k in [i, j-1] have been decremented.
    # Characters s[k] for k >= j (if any) remain untouched.
        
    return "".join(chars)