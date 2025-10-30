class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        from collections import defaultdict
        
        # Step 1: Count the frequency of each character in the string
        char_count = defaultdict(int)
        for char in s:
            char_count[char] += 1
        
        # Step 2: Identify special substrings
        special_substrings = []
        visited = set()
        
        i = 0
        while i < len(s):
            if s[i] in visited:
                i += 1
                continue
            
            # Start a new special substring
            start = i
            current_chars = set()
            
            while i < len(s) and s[i] not in visited:
                current_chars.add(s[i])
                i += 1
            
            # Check if the substring is special
            if all(char_count[char] == 1 for char in current_chars):
                special_substrings.append(s[start:i])
                visited.update(current_chars)
            else:
                # If it's not special, we can't use this substring
                visited.update(current_chars)
        
        # Step 3: Check if we can select k disjoint special substrings
        return len(special_substrings) >= k

# Example usage:
# sol = Solution()
# print(sol.maxSubstringLength("abcdbaefab", 2))  # Output: True
# print(sol.maxSubstringLength("cdefdc", 3))      # Output: False
# print(sol.maxSubstringLength("abeabe", 0))      # Output: True