import bisect

class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        parts = p.split('*')
        a, b, c = parts
        len_a, len_b, len_c = len(a), len(b), len(c)
        
        if len_a == 0 and len_b == 0 and len_c == 0:
            return 0
        
        if len_b == 0:
            if len_a == 0 and len_c == 0:
                return 0
            elif len_a == 0:
                if len_c == 0:
                    return 0
                else:
                    if c in s:
                        return len_c
                    else:
                        return -1
            elif len_c == 0:
                if a in s:
                    return len_a
                else:
                    return -1
            else:
                prefix_starts = []
                for i in range(len(s) - len_a + 1):
                    if s[i:i+len_a] == a:
                        prefix_starts.append(i)
                suffix_starts = []
                for i in range(len(s) - len_c + 1):
                    if s[i:i+len_c] == c:
                        suffix_starts.append(i)
                suffix_starts.sort()
                min_len = float('inf')
                if not prefix_starts or not suffix_starts:
                    return -1
                for i in prefix_starts:
                    pos = bisect.bisect_left(suffix_starts, i + len_a)
                    if pos < len(suffix_starts):
                        j_start = suffix_starts[pos]
                        L = j_start - i + len_c
                        if L < min_len:
                            min_len = L
                return min_len if min_len != float('inf') else -1
        else:
            prefix_starts = []
            if len_a > 0:
                for i in range(len(s) - len_a + 1):
                    if s[i:i+len_a] == a:
                        prefix_starts.append(i)
            mid_starts = []
            for i in range(len(s) - len_b + 1):
                if s[i:i+len_b] == b:
                    mid_starts.append(i)
            suffix_starts = []
            if len_c > 0:
                for i in range(len(s) - len_c + 1):
                    if s[i:i+len_c] == c:
                        suffix_starts.append(i)
            prefix_starts.sort()
            suffix_starts.sort()
            min_len = float('inf')
            if not mid_starts:
                return -1
            for k in mid_starts:
                if len_a == 0:
                    i_val = k
                else:
                    pos = bisect.bisect_right(prefix_starts, k - len_a) - 1
                    if pos < 0:
                        continue
                    i_val = prefix_starts[pos]
                if len_c == 0:
                    j_val = k + len_b - 1
                else:
                    pos2 = bisect.bisect_left(suffix_starts, k + len_b)
                    if pos2 == len(suffix_starts):
                        continue
                    j_start_val = suffix_starts[pos2]
                    j_val = j_start_val + len_c - 1
                L = j_val - i_val + 1
                if L < min_len:
                    min_len = L
            return min_len if min_len != float('inf') else -1