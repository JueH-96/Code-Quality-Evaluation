class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        n = len(word1)
        freq2_arr = [0] * 26
        for char in word2:
            idx = ord(char) - ord('a')
            freq2_arr[idx] += 1
        
        total_deficit = sum(freq2_arr)
        freq = [0] * 26
        l = 0
        total_valid = 0
        
        for r in range(n):
            idx_r = ord(word1[r]) - ord('a')
            if freq[idx_r] < freq2_arr[idx_r]:
                total_deficit -= 1
            freq[idx_r] += 1
            
            while total_deficit == 0 and l <= r:
                total_valid += (n - r)
                
                idx_l = ord(word1[l]) - ord('a')
                if freq[idx_l] <= freq2_arr[idx_l]:
                    total_deficit += 1
                freq[idx_l] -= 1
                l += 1
                
        return total_valid