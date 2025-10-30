class Solution:
    def findValidPair(self, s: str) -> str:
        from collections import defaultdict
        
        # Count the frequency of each digit in the string
        freq = defaultdict(int)
        for digit in s:
            freq[digit] += 1
        
        # Iterate through the string to find the first valid adjacent pair
        for i in range(len(s) - 1):
            first = s[i]
            second = s[i+1]
            if first != second:
                # Check if the counts match their numeric values
                if int(first) == freq[first] and int(second) == freq[second]:
                    return first + second
        return ""