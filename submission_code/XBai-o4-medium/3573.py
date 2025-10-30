from collections import defaultdict

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        req = defaultdict(int)
        for c in word2:
            req[c] += 1
        
        total_unique = len(req)
        if total_unique == 0:
            return 0
        
        count = defaultdict(int)
        required_met = 0
        left = 0
        result = 0
        
        for right in range(len(word1)):
            c = word1[right]
            if c in req:
                count[c] += 1
                # Check if this character's count just met the required
                if count[c] == req[c]:
                    required_met += 1
            
            # Try to move left as far as possible while the window is valid
            while required_met == total_unique:
                left_char = word1[left]
                if left_char in req:
                    count[left_char] -= 1
                    # Check if this character's count fell below required
                    if count[left_char] == req[left_char] - 1:
                        required_met -= 1
                left += 1
            
            # Add the current left to the result
            result += left
        
        return result