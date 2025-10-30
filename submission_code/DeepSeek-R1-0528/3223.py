class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        segments = []
        n = len(word)
        start = 0
        for i in range(1, n):
            if abs(ord(word[i]) - ord(word[i-1])) > 2:
                segments.append(word[start:i])
                start = i
        segments.append(word[start:])
        
        total = 0
        for seg in segments:
            n_seg = len(seg)
            if n_seg < k:
                continue
            res_seg = 0
            max_d = min(26, n_seg // k)
            for d in range(1, max_d + 1):
                L = d * k
                if L > n_seg:
                    break
                freq = [0] * 26
                distinct = 0
                count_k = 0
                for j in range(L):
                    char = seg[j]
                    idx = ord(char) - ord('a')
                    if freq[idx] == 0:
                        distinct += 1
                    if freq[idx] == k:
                        count_k -= 1
                    freq[idx] += 1
                    if freq[idx] == k:
                        count_k += 1
                if distinct == d and count_k == d:
                    res_seg += 1
                for i in range(1, n_seg - L + 1):
                    char_remove = seg[i-1]
                    idx_rm = ord(char_remove) - ord('a')
                    if freq[idx_rm] == k:
                        count_k -= 1
                    freq[idx_rm] -= 1
                    if freq[idx_rm] == 0:
                        distinct -= 1
                    elif freq[idx_rm] == k:
                        count_k += 1
                    char_add = seg[i+L-1]
                    idx_add = ord(char_add) - ord('a')
                    if freq[idx_add] == 0:
                        distinct += 1
                    if freq[idx_add] == k:
                        count_k -= 1
                    freq[idx_add] += 1
                    if freq[idx_add] == k:
                        count_k += 1
                    if distinct == d and count_k == d:
                        res_seg += 1
            total += res_seg
        return total