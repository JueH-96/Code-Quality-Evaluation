class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        from collections import defaultdict
        if not s:
            return ""
        freq = defaultdict(int)
        for char in s:
            freq[char] += 1
        T = max(freq.values())
        count = defaultdict(int)
        T_th_index = {}
        for idx, char in enumerate(s):
            count[char] += 1
            if count[char] == T:
                T_th_index[char] = idx
        arr = []
        for char, idx in T_th_index.items():
            arr.append((idx, char))
        arr.sort()
        return ''.join(char for idx, char in arr)