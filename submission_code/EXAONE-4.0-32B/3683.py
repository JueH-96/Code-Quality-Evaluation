class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        if numFriends == 1:
            return word
        
        n = len(word)
        max_len = n - (numFriends - 1)
        
        base1, base2 = 131, 131
        mod1, mod2 = 10**9+7, 10**9+9
        h1 = [0] * (n + 1)
        h2 = [0] * (n + 1)
        pow1 = [1] * (n + 1)
        pow2 = [1] * (n + 1)
        
        for i in range(1, n + 1):
            c_val = ord(word[i - 1]) - ord('a') + 1
            h1[i] = (h1[i - 1] * base1 + c_val) % mod1
            h2[i] = (h2[i - 1] * base2 + c_val) % mod2
            pow1[i] = (pow1[i - 1] * base1) % mod1
            pow2[i] = (pow2[i - 1] * base2) % mod2
            
        def get_hash(l, r):
            hash1 = (h1[r] - h1[l] * pow1[r - l]) % mod1
            hash2 = (h2[r] - h2[l] * pow2[r - l]) % mod2
            if hash1 < 0:
                hash1 += mod1
            if hash2 < 0:
                hash2 += mod2
            return (hash1, hash2)
        
        best_start = 0
        best_len = min(max_len, n)
        
        for i in range(1, n):
            L_i = min(max_len, n - i)
            min_len = min(best_len, L_i)
            
            low, high = 0, min_len
            lcp = 0
            while low <= high:
                mid = (low + high) // 2
                hash_best = get_hash(best_start, best_start + mid)
                hash_i = get_hash(i, i + mid)
                if hash_best == hash_i:
                    lcp = mid
                    low = mid + 1
                else:
                    high = mid - 1
                    
            if lcp == min_len:
                if best_len < L_i:
                    best_start = i
                    best_len = L_i
            else:
                if word[best_start + lcp] < word[i + lcp]:
                    best_start = i
                    best_len = L_i
                    
        return word[best_start:best_start + best_len]