from collections import defaultdict

class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def split_into_runs(s):
            if not s:
                return []
            runs = []
            start = 0
            for i in range(1, len(s)):
                if abs(ord(s[i]) - ord(s[i-1])) > 2:
                    runs.append(s[start:i])
                    start = i
            runs.append(s[start:])
            return runs
        
        runs = split_into_runs(word)
        total = 0
        
        for run in runs:
            n = len(run)
            for m in range(1, 27):
                L = m * k
                if L > n:
                    continue
                freq = defaultdict(int)
                distinct = 0
                valid = 0
                for i in range(L):
                    c = run[i]
                    freq[c] += 1
                    if freq[c] == 1:
                        distinct += 1
                    if freq[c] == k:
                        valid += 1
                    elif freq[c] == k + 1:
                        valid -= 1
                if distinct == m and valid == m:
                    total += 1
                for i in range(L, n):
                    left_char = run[i - L]
                    if freq[left_char] == k:
                        valid -= 1
                    freq[left_char] -= 1
                    if freq[left_char] == 0:
                        distinct -= 1
                    if freq[left_char] == k - 1:
                        valid += 1
                    right_char = run[i]
                    if freq[right_char] == 0:
                        distinct += 1
                    freq[right_char] += 1
                    if freq[right_char] == k:
                        valid += 1
                    elif freq[right_char] == k + 1:
                        valid -= 1
                    if distinct == m and valid == m:
                        total += 1
        return total