import bisect

class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        def build_lps(pattern):
            m = len(pattern)
            lps = [0] * m
            length = 0
            i = 1
            while i < m:
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length-1]
                    else:
                        lps[i] = 0
                        i += 1
            return lps
        
        def find_occurrences(text, pattern):
            n = len(text)
            m = len(pattern)
            if m == 0:
                return list(range(n + 1))  # All possible starts including n
            lps = build_lps(pattern)
            occurrences = []
            i = 0
            j = 0
            while i < n:
                if text[i] == pattern[j]:
                    i += 1
                    j += 1
                    if j == m:
                        occurrences.append(i - j)
                        j = lps[j-1]
                else:
                    if j != 0:
                        j = lps[j-1]
                    else:
                        i += 1
            return occurrences
        
        # Split the pattern into L, M, R
        stars = [i for i, c in enumerate(p) if c == '*']
        first_star = stars[0]
        second_star = stars[1]
        L = p[:first_star]
        M = p[first_star+1: second_star]
        R = p[second_star+1:]
        
        # Find occurrences of each part in s
        list_L = find_occurrences(s, L)
        list_M = find_occurrences(s, M)
        list_R = find_occurrences(s, R)
        
        # Check if required non-empty parts have occurrences
        if (len(L) > 0 and not list_L) or (len(M) > 0 and not list_M) or (len(R) > 0 and not list_R):
            return -1
        
        min_len = float('inf')
        l_len = len(L)
        m_len = len(M)
        r_len = len(R)
        
        # Iterate through all possible starts of L
        for i in list_L:
            # Find the first valid M occurrence after L
            low_j = i + l_len
            j_idx = bisect.bisect_left(list_M, low_j)
            if j_idx >= len(list_M):
                continue
            j = list_M[j_idx]
            
            # Find the first valid R occurrence after M
            low_k = j + m_len
            k_idx = bisect.bisect_left(list_R, low_k)
            if k_idx >= len(list_R):
                continue
            k = list_R[k_idx]
            
            current_len = (k + r_len) - i
            if current_len < min_len:
                min_len = current_len
        
        return min_len if min_len != float('inf') else -1