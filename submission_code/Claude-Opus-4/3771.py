class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        if k == 0:
            return True
        
        n = len(s)
        
        # Count frequency of each character
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        
        # For each character, find its first and last occurrence
        first_occurrence = {}
        last_occurrence = {}
        
        for i in range(n):
            c = s[i]
            if c not in first_occurrence:
                first_occurrence[c] = i
            last_occurrence[c] = i
        
        # Find all maximal special substrings
        # A substring is special if all its characters don't appear outside it
        special_count = 0
        i = 0
        
        while i < n:
            # Start a potential special substring
            start = i
            end = i
            chars_in_substring = set()
            
            # Expand the substring to include all characters that must be together
            while i <= end and i < n:
                c = s[i]
                chars_in_substring.add(c)
                # Update end to include the entire range of this character
                end = max(end, last_occurrence[c])
                i += 1
            
            # Check if this substring is valid (all chars appear only in this range)
            is_valid = True
            for c in chars_in_substring:
                if first_occurrence[c] < start or last_occurrence[c] > end:
                    is_valid = False
                    break
            
            # Also check that it's not the entire string
            if is_valid and not (start == 0 and end == n - 1):
                special_count += 1
        
        return special_count >= k