class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        def try_exact_match():
            result = []
            w1_idx = 0
            
            for w2_idx in range(len(word2)):
                found = False
                while w1_idx < len(word1):
                    if word1[w1_idx] == word2[w2_idx]:
                        result.append(w1_idx)
                        w1_idx += 1
                        found = True
                        break
                    w1_idx += 1
                
                if not found:
                    return None
            
            return result
        
        def try_with_one_change(change_pos):
            result = []
            w1_idx = 0
            
            # Build prefix with exact matches
            for w2_idx in range(change_pos):
                found = False
                while w1_idx < len(word1):
                    if word1[w1_idx] == word2[w2_idx]:
                        result.append(w1_idx)
                        w1_idx += 1
                        found = True
                        break
                    w1_idx += 1
                
                if not found:
                    return None
            
            # Use one change at change_pos
            if w1_idx < len(word1):
                result.append(w1_idx)
                w1_idx += 1
            else:
                return None
            
            # Build suffix with exact matches
            for w2_idx in range(change_pos + 1, len(word2)):
                found = False
                while w1_idx < len(word1):
                    if word1[w1_idx] == word2[w2_idx]:
                        result.append(w1_idx)
                        w1_idx += 1
                        found = True
                        break
                    w1_idx += 1
                
                if not found:
                    return None
            
            return result
        
        # Try exact match first
        result = try_exact_match()
        if result is not None:
            return result
        
        # Try with one change at each position
        for change_pos in range(len(word2)):
            result = try_with_one_change(change_pos)
            if result is not None:
                return result
        
        return []