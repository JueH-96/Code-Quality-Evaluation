class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n = len(word2)
        m = len(word1)
        if n > m:
            return []
        
        # To find the lexicographically smallest sequence, we need to choose the smallest possible indices
        # that can form a sequence almost equal to word2.
        
        # We will iterate through word2 and try to find the smallest possible index in word1 that matches or can be changed to match.
        
        result = []
        used = set()
        mismatch_count = 0
        last_used = -1
        
        for i in range(n):
            target_char = word2[i]
            # Find the smallest index in word1 that is not used and matches or can be changed to target_char
            # and is greater than last_used
            found = False
            for j in range(last_used + 1, m):
                if j not in used:
                    if word1[j] == target_char:
                        result.append(j)
                        used.add(j)
                        last_used = j
                        found = True
                        break
                    elif mismatch_count == 0:
                        # We can change this character
                        result.append(j)
                        used.add(j)
                        last_used = j
                        mismatch_count += 1
                        found = True
                        break
            if not found:
                return []
        
        # Now, check if the sequence is valid
        # The sequence must be such that the concatenated string is almost equal to word2
        # i.e., at most one character is different
        # Since we have already ensured that during the selection, we can return the result
        return result