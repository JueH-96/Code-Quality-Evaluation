class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        segments = []
        i = 0
        while i < n:
            j = i + 1
            while j < n and abs(ord(word[j]) - ord(word[j-1])) <= 2:
                j += 1
            segments.append(word[i:j])
            i = j
        
        ans = 0
        for seg in segments:
            ans += self.count_in_segment(seg, k)
        return ans

    def count_in_segment(self, segment, k):
        L = len(segment)
        ans = 0
        for d in range(1, 27):
            L0 = d * k
            if L0 > L:
                break
            freq = [0] * 26
            for i in range(L0):
                c = segment[i]
                idx = ord(c) - ord('a')
                freq[idx] += 1
            
            distinct = 0
            count_bad = 0
            for i in range(26):
                if freq[i] > 0:
                    distinct += 1
                    if freq[i] != k:
                        count_bad += 1
            if distinct == d and count_bad == 0:
                ans += 1
            
            for i in range(L0, L):
                c_out = segment[i - L0]
                idx_out = ord(c_out) - ord('a')
                freq[idx_out] -= 1
                
                c_in = segment[i]
                idx_in = ord(c_in) - ord('a')
                freq[idx_in] += 1
                
                distinct = 0
                count_bad = 0
                for j in range(26):
                    if freq[j] > 0:
                        distinct += 1
                        if freq[j] != k:
                            count_bad += 1
                if distinct == d and count_bad == 0:
                    ans += 1
        return ans