class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        # Convert the string to a list of characters to allow mutation.
        chars = list(s)
        
        # Initialize two pointers, one at the start and one at the end of the list.
        left, right = 0, len(s) - 1
        
        # Iterate from both ends towards the center.
        while left < right:
            # If the two characters are not the same, we need to make a change.
            if chars[left] != chars[right]:
                # To make the resulting palindrome lexicographically smallest,
                # we must choose the smaller of the two characters for both positions.
                # This ensures the character at the `left` index (which has a
                # greater impact on lexicographical order) is as small as possible.
                # This also guarantees the minimum number of operations (one change per
                # mismatched pair).
                if chars[left] < chars[right]:
                    chars[right] = chars[left]
                else:
                    chars[left] = chars[right]
            
            # Move the pointers inward to process the next pair of characters.
            left += 1
            right -= 1
            
        # Join the list of characters back into a string and return it.
        return "".join(chars)