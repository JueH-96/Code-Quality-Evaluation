class Solution:
    def maximumLength(self, s: str) -> int:
        n = len(s)
        count = {}
        
        # Count consecutive occurrences of each character
        for i in range(n):
            char = s[i]
            j = i
            while j < n and s[j] == char:
                length = j - i + 1
                count[char] = count.get(char, {})
                count[char][length] = count[char].get(length, 0) + 1
                j += 1
        
        max_length = -1
        
        # Check for each character and length
        for char in count:
            for length in sorted(count[char].keys(), reverse=True):
                occurrences = count[char][length]
                if occurrences >= 3:
                    return length
                elif occurrences == 2 and length > 1:
                    max_length = max(max_length, length - 1)
                elif occurrences == 1 and length > 2:
                    max_length = max(max_length, length - 2)
        
        return max_length