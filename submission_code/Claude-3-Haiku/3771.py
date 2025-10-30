class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        n = len(s)
        
        # Create a dictionary to store the count of each character in the string
        char_count = {}
        for char in s:
            char_count[char] = char_count.get(char, 0) + 1
        
        # Sort the characters based on their count in descending order
        sorted_chars = sorted(char_count.items(), key=lambda x: x[1], reverse=True)
        
        # Check if there are more than k unique characters
        if len(sorted_chars) > k:
            return False
        
        # Check if it is possible to select k disjoint special substrings
        start = 0
        end = 0
        while end < n:
            # Expand the window to find a special substring
            while end < n and all(count <= 1 for _, count in sorted_chars):
                end += 1
            
            # If we found a special substring, check if we can select k of them
            if end - start >= n // k:
                return True
            
            # Otherwise, move the start pointer to the right
            while start < end and any(count > 1 for _, count in sorted_chars):
                char_count[s[start]] -= 1
                start += 1
        
        # If we couldn't find k disjoint special substrings, return False
        return False