class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n1, n2 = len(word1), len(word2)
        
        # First, let's find the rightmost possible position for each character in word2
        # This helps us know if we can complete the sequence from any given position
        rightmost = [-1] * n2
        j = n1 - 1
        for i in range(n2 - 1, -1, -1):
            while j >= 0 and word1[j] != word2[i]:
                j -= 1
            if j < 0:
                break
            rightmost[i] = j
            j -= 1
        
        result = []
        start = 0
        used_change = False
        
        for i in range(n2):
            found = False
            
            # Try to find a matching character
            for j in range(start, n1):
                if word1[j] == word2[i]:
                    # Check if we can complete the rest of the sequence
                    can_complete = True
                    if i < n2 - 1:
                        # We need to check if we can match the remaining characters
                        # considering we might have already used our change
                        if used_change:
                            # Must match exactly from here
                            can_complete = rightmost[i+1] > j if i+1 < n2 else True
                        else:
                            # Can use one change for the remaining
                            can_complete = True
                            # But we need at least n2-i-1 characters left
                            if n1 - j - 1 < n2 - i - 1:
                                can_complete = False
                    
                    if can_complete:
                        result.append(j)
                        start = j + 1
                        found = True
                        break
            
            if not found and not used_change:
                # Try using our one allowed change
                for j in range(start, n1):
                    # Check if we can complete the rest after using the change
                    can_complete = True
                    if i < n2 - 1:
                        # Must match exactly from here
                        can_complete = rightmost[i+1] > j if i+1 < n2 else True
                    
                    if can_complete:
                        result.append(j)
                        start = j + 1
                        used_change = True
                        found = True
                        break
            
            if not found:
                return []
        
        return result