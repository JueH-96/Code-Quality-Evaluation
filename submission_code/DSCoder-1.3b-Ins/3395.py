class Solution:
    def minAnagramLength(self, s: str) -> int:
        # Create a dictionary to store the frequency of each character in the string
        freq = {}
        for char in s:
            if char in freq:
                freq[char] += 1
            else:
                freq[char] = 1
        
        # Count the number of characters that appear an odd number of times
        odd_count = 0
        for count in freq.values():
            if count % 2 != 0:
                odd_count += 1
        
        # The minimum length of the string t is the number of characters that appear an odd number of times plus one
        return odd_count + 1