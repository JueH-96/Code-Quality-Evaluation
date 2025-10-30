class Solution:
  def canMakeSubsequence(self, str1: str, str2: str) -> bool:
    n = len(str1)
    m = len(str2)

    # According to constraints: 1 <= str1.length, 1 <= str2.length.
    # This means n >= 1 and m >= 1.

    # Helper function to get the next character in cyclical order ('z' -> 'a').
    # Defined locally as it's only used here.
    def get_next_char(c: str) -> str:
        if c == 'z':
            return 'a'
        # For 'a' through 'y', the next character is chr(ord(c) + 1).
        return chr(ord(c) + 1)

    p1 = 0  # Pointer for str1
    p2 = 0  # Pointer for str2

    # Iterate while both pointers are within their respective string bounds.
    while p1 < n and p2 < m:
        # Current character from str1 and str2.
        s1_char = str1[p1]
        s2_char = str2[p2]

        # Check if str1[p1] can form str2[p2].
        # This can be true if str1[p1] is already str2[p2],
        # OR if str1[p1] incremented becomes str2[p2].
        # This covers the "at most one operation" requirement:
        # - If s1_char == s2_char, this character str1[p1] is used as-is (not part of the "increment set").
        # - If get_next_char(s1_char) == s2_char, this character str1[p1] is used after incrementing
        #   (it's part of the "increment set").
        # The greedy strategy tries to match str2[p2] with the earliest possible str1[p1].
        
        if s1_char == s2_char or get_next_char(s1_char) == s2_char:
            # Match found for str2[p2]. Advance p2 to look for the next character of str2.
            p2 += 1
        
        # Always advance p1.
        # - If str1[p1] was used to match str2[p2], we need to look for str2[p2+1] in str1[p1+1:].
        # - If str1[p1] was not used (could not match str2[p2]), we need to look for str2[p2] in str1[p1+1:].
        # In both cases, str1[p1] is "consumed", so p1 moves to the next character in str1.
        p1 += 1
            
    # If p2 reached m, it means all m characters of str2 were successfully matched.
    # Otherwise, str2 could not be formed as a subsequence.
    return p2 == m