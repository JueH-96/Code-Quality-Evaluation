class Solution:
    def findValidPair(self, s: str) -> str:
        # Count frequency of each digit
        freq = {}
        for digit in s:
            freq[digit] = freq.get(digit, 0) + 1
        
        # Check each adjacent pair
        for i in range(len(s) - 1):
            # Get the two adjacent digits
            first = s[i]
            second = s[i + 1]
            
            # Skip if digits are same
            if first == second:
                continue
            
            # Convert digits to integers for comparison
            first_val = int(first)
            second_val = int(second)
            
            # Check if frequency matches the digit value
            if freq[first] == first_val and freq[second] == second_val:
                return first + second
        
        # No valid pair found
        return ""