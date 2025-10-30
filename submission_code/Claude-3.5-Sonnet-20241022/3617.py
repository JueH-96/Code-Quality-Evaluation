class Solution:
    def possibleStringCount(self, word: str) -> int:
        n = len(word)
        result = set()
        
        # Add the original string as one possibility
        result.add(word)
        
        # Find all groups of consecutive same characters
        i = 0
        while i < n:
            # Count consecutive same characters
            count = 1
            while i + count < n and word[i] == word[i + count]:
                count += 1
            
            # If we found consecutive same characters
            if count > 1:
                # Try removing 1 to count-1 characters from this group
                for j in range(1, count):
                    # Create new string with j characters removed
                    new_string = word[:i] + word[i:i+count-j] + word[i+count:]
                    result.add(new_string)
            
            i += count
        
        return len(result)