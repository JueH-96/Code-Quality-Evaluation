class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        n = len(word1)
        required = [0] * 26
        for c in word2:
            idx = ord(c) - ord('a')
            required[idx] += 1
        
        total_required = 0
        for count in required:
            if count > 0:
                total_required += 1
        
        curr = [0] * 26
        satisfied = 0
        r = 0
        ans = 0
        for l in range(n):
            while r < n and satisfied < total_required:
                idx_r = ord(word1[r]) - ord('a')
                curr[idx_r] += 1
                if required[idx_r] > 0 and curr[idx_r] == required[idx_r]:
                    satisfied += 1
                r += 1
            
            if satisfied == total_required:
                ans += (n - r + 1)
            
            idx_l = ord(word1[l]) - ord('a')
            if required[idx_l] > 0 and curr[idx_l] == required[idx_l]:
                satisfied -= 1
            curr[idx_l] -= 1
        
        return ans