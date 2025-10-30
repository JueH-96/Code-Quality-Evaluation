import bisect

class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        
        def kmp_find_all(text: str, pattern: str) -> list[int]:
            """
            Finds all starting indices of a pattern in a text using the KMP algorithm.
            """
            n, m = len(text), len(pattern)
            if m == 0:
                return []
            
            # Build LPS array for KMP
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
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            
            # KMP search
            indices = []
            i, j = 0, 0  # i for text, j for pattern
            while i < n:
                if pattern[j] == text[i]:
                    i += 1
                    j += 1
                
                if j == m:
                    indices.append(i - j)
                    j = lps[j - 1]
                elif i < n and pattern[j] != text[i]:
                    if j != 0:
                        j = lps[j - 1]
                    else:
                        i += 1
            return indices

        # 1. Parse the pattern
        parts = p.split('*')
        prefix, middle, suffix = parts[0], parts[1], parts[2]
        L_pref, L_mid, L_suff = len(prefix), len(middle), len(suffix)

        # 2. Find all occurrences of each part
        prefix_starts = kmp_find_all(s, prefix)
        if L_pref > 0 and not prefix_starts:
            return -1

        middle_starts = kmp_find_all(s, middle)
        if L_mid > 0 and not middle_starts:
            return -1

        suffix_starts = kmp_find_all(s, suffix)
        if L_suff > 0 and not suffix_starts:
            return -1
            
        min_len = float('inf')

        # 3. Handle the simpler case where the middle part is empty
        if L_mid == 0:
            if L_pref == 0 and L_suff == 0:  # p = "**"
                return 0
            if L_pref == 0:  # p = "**suffix"
                return L_suff
            if L_suff == 0:  # p = "prefix**"
                return L_pref
            
            # Case: p = "prefix**suffix"
            for i in prefix_starts:
                # Find the earliest suffix that starts after the prefix ends
                k_idx = bisect.bisect_left(suffix_starts, i + L_pref)
                if k_idx < len(suffix_starts):
                    k = suffix_starts[k_idx]
                    min_len = min(min_len, k + L_suff - i)
        else:
            # 4. Main logic for non-empty middle part
            for j in middle_starts:
                # Find the best possible prefix start `i` for this `j`
                i_candidate = -1
                if L_pref == 0:
                    i_candidate = j
                else:
                    # Find largest i in prefix_starts s.t. i <= j - L_pref
                    p_idx = bisect.bisect_right(prefix_starts, j - L_pref)
                    if p_idx > 0:
                        i_candidate = prefix_starts[p_idx - 1]
                
                if i_candidate == -1:
                    continue

                # Find the best possible suffix start `k` for this `j`
                k_candidate = -1
                if L_suff == 0:
                    k_candidate = j + L_mid
                else:
                    # Find smallest k in suffix_starts s.t. k >= j + L_mid
                    s_idx = bisect.bisect_left(suffix_starts, j + L_mid)
                    if s_idx < len(suffix_starts):
                        k_candidate = suffix_starts[s_idx]

                if k_candidate != -1:
                    length = (k_candidate + L_suff) - i_candidate
                    min_len = min(min_len, length)

        return min_len if min_len != float('inf') else -1