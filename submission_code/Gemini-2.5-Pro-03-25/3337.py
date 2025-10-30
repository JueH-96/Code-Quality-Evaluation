import math # Not strictly needed as we use integer arithmetic, but good practice if relevant math ops were used.

class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        """
        Counts the total number of substrings of s that start and end with the character c.

        Args:
            s: The input string. 1 <= s.length <= 10^5. Consists of lowercase English letters.
            c: The target character. A lowercase English letter.

        Returns:
            The total number of valid substrings. The result can be up to approximately 5 * 10^9.
        
        Approach:
        A substring is defined by its start and end indices. A substring starting at index i 
        and ending at index j (where i <= j) is valid if it starts with character c and ends 
        with character c. This means s[i] == c and s[j] == c.

        Let k be the total number of occurrences of character c in the string s.
        Consider the indices where c appears: idx_1, idx_2, ..., idx_k.
        Any pair of these indices (idx_p, idx_q) where p <= q corresponds uniquely to a 
        substring s[idx_p : idx_q + 1] that starts and ends with c.

        The problem reduces to counting the number of pairs of indices (p, q) such that 
        1 <= p <= k and p <= q <= k.
        
        This count can be found by considering each occurrence of c as a potential end point.
        - The first occurrence of c (at index idx_1) can form 1 valid substring (ending at idx_1, starting at idx_1).
        - The second occurrence of c (at index idx_2) can form 2 valid substrings (ending at idx_2, starting at idx_1 or idx_2).
        - ...
        - The k-th occurrence of c (at index idx_k) can form k valid substrings (ending at idx_k, starting at idx_1, idx_2, ..., idx_k).

        The total number of valid substrings is the sum 1 + 2 + ... + k.
        This is the sum of the first k positive integers, which has a well-known formula: k * (k + 1) / 2.

        Example: s = "abada", c = "a"
        'a' appears at indices 0, 2, 4. So, k = 3.
        Valid substrings:
        - Starting at index 0: "a" (ends at 0), "aba" (ends at 2), "abada" (ends at 4) -> 3 substrings
        - Starting at index 2: "a" (ends at 2), "ada" (ends at 4) -> 2 substrings
        - Starting at index 4: "a" (ends at 4) -> 1 substring
        This view counts based on starting index. Let's use the ending index view for the 1+2+...+k sum:
        - Substrings ending at index 0: "a" (starts at 0) -> 1 substring
        - Substrings ending at index 2: "aba" (starts at 0), "a" (starts at 2) -> 2 substrings
        - Substrings ending at index 4: "abada" (starts at 0), "ada" (starts at 2), "a" (starts at 4) -> 3 substrings
        Total = 1 + 2 + 3 = 6.
        Using the formula: k * (k + 1) // 2 = 3 * (3 + 1) // 2 = 3 * 4 // 2 = 12 // 2 = 6.

        Implementation Details:
        1. Count the occurrences of character c in the string s. The `s.count(c)` method is efficient for this.
        2. Apply the formula `k * (k + 1) // 2` where k is the count. Use integer division `//` to ensure the result is an integer.
        3. Python's built-in integers handle arbitrary precision, so overflow is not a concern for results up to 5 * 10^9.
        """
        
        # Count the number of times character 'c' appears in the string 's'.
        # Let this count be k.
        count_c = s.count(c)
        
        # The number of substrings starting and ending with 'c' is given by
        # the sum 1 + 2 + ... + k, which equals k * (k + 1) / 2.
        # If count_c is 0, the formula correctly yields 0 * 1 // 2 = 0.
        
        # Calculate the result using integer division //.
        result = count_c * (count_c + 1) // 2
        
        # Return the calculated total number of substrings.
        return result