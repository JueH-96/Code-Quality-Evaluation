class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        total_count = 0
        char_c_occurrences = 0
        
        # Iterate through each character in the string s
        for char in s:
            # If the current character is equal to the target character 'c'
            if char == c:
                # Increment the count of occurrences of 'c' found so far.
                # This 'c' is the (char_c_occurrences)-th 'c' found.
                char_c_occurrences += 1
                
                # Each time we find 'c', it can serve as the ending character
                # for any substring that started with a 'c' that appeared
                # at or before the current position.
                # The number of such starting 'c's is exactly 'char_c_occurrences'.
                # So, we add this number to our total count.
                total_count += char_c_occurrences
                
        return total_count