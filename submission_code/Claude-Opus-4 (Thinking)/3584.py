class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n1, n2 = len(word1), len(word2)
        
        # Precompute next occurrence of each character from each position
        next_occ = [[-1] * 26 for _ in range(n1 + 1)]
        for i in range(n1 - 1, -1, -1):
            for c in range(26):
                next_occ[i][c] = next_occ[i + 1][c]
            next_occ[i][ord(word1[i]) - ord('a')] = i
        
        def can_match_suffix(start1, start2):
            # Check if we can match word2[start2:] from word1[start1:] exactly
            pos = start1
            for j in range(start2, n2):
                char_idx = ord(word2[j]) - ord('a')
                if next_occ[pos][char_idx] == -1:
                    return False
                pos = next_occ[pos][char_idx] + 1
            return True
        
        result = []
        pos = 0
        change_used = False
        
        for i in range(n2):
            char_needed = word2[i]
            char_idx = ord(char_needed) - ord('a')
            
            # Option 1: exact match
            exact_match_pos = next_occ[pos][char_idx]
            
            # Option 2: use change (if not used and position available)
            change_pos = pos if pos < n1 and not change_used else -1
            
            # Check which options are valid
            use_exact = exact_match_pos != -1 and can_match_suffix(exact_match_pos + 1, i + 1)
            use_change = change_pos != -1 and can_match_suffix(change_pos + 1, i + 1)
            
            if not use_exact and not use_change:
                return []
            
            # Choose the option with smaller index
            if use_change and (not use_exact or change_pos < exact_match_pos):
                result.append(change_pos)
                pos = change_pos + 1
                change_used = True
            else:
                result.append(exact_match_pos)
                pos = exact_match_pos + 1
        
        return result