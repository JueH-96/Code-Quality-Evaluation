class Solution:
    def getSmallestString(self, s: str) -> str:
        s = list(s)
        n = len(s)
        swapped = False
        
        for i in range(n-1):
            # Check if current and next digits have same parity
            if (int(s[i]) % 2 == int(s[i+1]) % 2):
                # If next digit is smaller, swap them
                if s[i+1] < s[i]:
                    s[i], s[i+1] = s[i+1], s[i]
                    swapped = True
                    break
        
        # If no swap was needed, try swapping same parity digits to get lexicographically smallest
        if not swapped:
            for i in range(n-1):
                if (int(s[i]) % 2 == int(s[i+1]) % 2):
                    s[i], s[i+1] = s[i+1], s[i]
                    break
                    
        return ''.join(s)