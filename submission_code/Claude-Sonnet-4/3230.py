class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        if len(word) <= 1:
            return 0
        
        word = list(word)
        operations = 0
        
        def are_almost_equal(a, b):
            return a == b or abs(ord(a) - ord(b)) == 1
        
        for i in range(1, len(word)):
            if are_almost_equal(word[i-1], word[i]):
                operations += 1
                
                # Find a safe character to replace word[i]
                # Try characters that are far from both neighbors
                for c in 'aceg':  # Use characters with gaps
                    valid = True
                    
                    if are_almost_equal(c, word[i-1]):
                        valid = False
                    
                    if i + 1 < len(word) and are_almost_equal(c, word[i+1]):
                        valid = False
                    
                    if valid:
                        word[i] = c
                        break
                else:
                    # If none of the preferred characters work, try all
                    for c in 'abcdefghijklmnopqrstuvwxyz':
                        valid = True
                        
                        if are_almost_equal(c, word[i-1]):
                            valid = False
                        
                        if i + 1 < len(word) and are_almost_equal(c, word[i+1]):
                            valid = False
                        
                        if valid:
                            word[i] = c
                            break
        
        return operations