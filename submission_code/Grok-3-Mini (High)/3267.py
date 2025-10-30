class Solution:
    def maximumLength(self, s: str) -> int:
        runs = {}
        i = 0
        max_run_len = 0
        while i < len(s):
            cur_char = s[i]
            count = 1
            i += 1
            while i < len(s) and s[i] == cur_char:
                count += 1
                i += 1
            if cur_char not in runs:
                runs[cur_char] = []
            runs[cur_char].append(count)
            max_run_len = max(max_run_len, count)
        
        for K in range(max_run_len, 0, -1):
            for lens in runs.values():
                sum_occ = sum(L - K + 1 for L in lens if L >= K)
                if sum_occ >= 3:
                    return K
        
        return -1