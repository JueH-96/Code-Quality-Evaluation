class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        len_word1 = len(word1)
        len_word2 = len(word2)
        
        if len_word2 > len_word1:
            return 0
        
        # Compute required character counts for word2
        required = [0] * 26
        for c in word2:
            idx = ord(c) - ord('a')
            required[idx] += 1
        
        # Compute prefix sum array for word1
        prefix = [ [0] * 26 ]
        for c in word1:
            new_row = prefix[-1].copy()
            idx = ord(c) - ord('a')
            new_row[idx] += 1
            prefix.append(new_row)
        
        ans = 0
        
        for i in range(len_word1):
            len_w2 = len_word2
            min_j = i + len_w2 - 1
            if min_j >= len_word1:
                continue  # Not enough length to have a valid substring
            
            low = min_j
            high = len_word1 - 1
            result_j = None
            
            while low <= high:
                mid = (low + high) // 2
                valid = True
                
                for c in range(26):
                    if required[c] == 0:
                        continue
                    if prefix[mid + 1][c] - prefix[i][c] < required[c]:
                        valid = False
                        break
                
                if valid:
                    result_j = mid
                    high = mid - 1
                else:
                    low = mid + 1
            
            if result_j is not None:
                ans += (len_word1 - result_j)
        
        return ans