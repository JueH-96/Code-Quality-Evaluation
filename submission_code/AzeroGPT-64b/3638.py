from collections import Counter
from math import inf

class Solution:
    def makeStringGood(self, s: str) -> int:
        def helper(recent_char, frequent_limit):
            if not s: return 0
            first, *rest = s
            if first == recent_char:
                return 1 + helper(recent_char, frequent_limit)
            else:
                if first.islower() and ord(first) - ord('a') == frequent_limit - 1:
                    return 1 + helper(chr(ord('a') + frequent_limit - 1), frequent_limit - 1)
                else:
                    return min(
                        (helper(recent_char, frequent_limit), helper(first, frequent_limit)) if frequent_limit else 
                        (helper(recent_char, frequent_limit), 
                         1 + helper(chr(ord('a') + frequent_limit), frequent_limit - 1))  
                    )
        
        s = 'a' + s + 'a'
        char_counts = list(Counter(s[1:-1]).values())
        char_counts.sort()
        # Calculate the operations for different frequent limits
        operations = [helper(s[0], freq) for freq in range(len(char_counts), 0, -1)]
        # Return the minimum number of operations to make the string good
        return min(operations)