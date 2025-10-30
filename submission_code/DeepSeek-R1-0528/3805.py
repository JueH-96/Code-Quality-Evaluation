class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        total_ones_original = s.count('1')
        t = "1" + s + "1"
        segments = []
        i, n = 0, len(t)
        while i < n:
            j = i
            while j < n and t[j] == t[i]:
                j += 1
            segments.append((t[i], j - i))
            i = j
        
        global_zeros = [length for char, length in segments if char == '0']
        
        if not global_zeros:
            return total_ones_original
        
        max_val = 0
        second_max = 0
        freq_max = 0
        for x in global_zeros:
            if x > max_val:
                second_max = max_val
                max_val = x
                freq_max = 1
            elif x == max_val:
                freq_max += 1
            elif x > second_max:
                second_max = x
        
        best = total_ones_original
        n_segs = len(segments)
        for i in range(2, n_segs - 1, 2):
            A = segments[i-1][1]
            B = segments[i+1][1]
            L = segments[i][1]
            new_zeros = A + L + B
            count_occur = 0
            if A == max_val:
                count_occur += 1
            if B == max_val:
                count_occur += 1
            if freq_max - count_occur > 0:
                max_excluding = max_val
            else:
                max_excluding = second_max
            M = max(new_zeros, max_excluding)
            candidate_result = total_ones_original - L + M
            if candidate_result > best:
                best = candidate_result
        
        return best