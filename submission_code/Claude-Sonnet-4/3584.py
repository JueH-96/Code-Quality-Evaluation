class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n1, n2 = len(word1), len(word2)
        
        # Precompute suffix matching: suffix[i] = max j such that 
        # word1[i:] can match word2[j:] exactly (without changes)
        suffix = [n2] * (n1 + 1)  # suffix[n1] = n2 (empty suffix matches empty)
        
        j = n2 - 1
        for i in range(n1 - 1, -1, -1):
            suffix[i] = suffix[i + 1]  # inherit from next position
            if j >= 0 and word1[i] == word2[j]:
                suffix[i] = j
                j -= 1
        
        result = []
        i = 0  # pointer for word1
        j = 0  # pointer for word2
        used_change = False
        
        while j < n2:
            if i >= n1:
                return []  # ran out of characters in word1
            
            if word1[i] == word2[j]:
                # Characters match, add to result
                result.append(i)
                i += 1
                j += 1
            elif not used_change:
                # Try using the change here
                # Check if we can match the remaining suffix after this change
                if suffix[i + 1] <= j + 1:
                    # We can match the remaining characters
                    result.append(i)
                    used_change = True
                    i += 1
                    j += 1
                else:
                    # Can't use change here, move to next character in word1
                    i += 1
            else:
                # Already used change and characters don't match
                i += 1
        
        return result if len(result) == n2 else []