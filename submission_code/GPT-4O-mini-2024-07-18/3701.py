class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        if n < 3:
            return ""
        
        # To store the result characters
        result = []
        
        # Count the frequency of each character
        count = {}
        for char in caption:
            if char in count:
                count[char] += 1
            else:
                count[char] = 1
        
        # Create a list of characters sorted lexicographically
        sorted_chars = sorted(count.keys())
        
        # Create a new list to build the good caption
        i = 0
        while i < n:
            char = caption[i]
            # Count how many of the same character are in a row
            j = i
            while j < n and caption[j] == char:
                j += 1
            length = j - i
            
            # If the length is less than 3, we need to change some characters
            if length < 3:
                # Calculate how many we need to change
                needed = 3 - length
                # Try to change to the previous character if possible
                if char > 'a':
                    prev_char = chr(ord(char) - 1)
                    if prev_char in count:
                        # Change to previous character
                        result.append(prev_char * 3)
                        count[prev_char] += 3
                    else:
                        # If we can't change to previous, we can only change to next
                        result.append(char * 3)
                        count[char] += 3
                else:
                    # If we are at 'a', we can only change to next
                    next_char = chr(ord(char) + 1)
                    result.append(next_char * 3)
                    count[next_char] += 3
            else:
                # If we have 3 or more, just append it
                result.append(char * length)
            
            # Move to the next different character
            i = j
        
        # Join the result and check if it is a good caption
        final_caption = ''.join(result)
        
        # Check if the final caption is a good caption
        if len(final_caption) < 3:
            return ""
        
        # Verify groups of characters
        for k in range(0, len(final_caption), 3):
            if k + 2 >= len(final_caption) or final_caption[k] != final_caption[k + 1] or final_caption[k] != final_caption[k + 2]:
                return ""
        
        return final_caption