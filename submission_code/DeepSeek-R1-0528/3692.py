import bisect

class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        n = len(s)
        stars = []
        for i, char in enumerate(p):
            if char == '*':
                stars.append(i)
        A = p[:stars[0]]
        B = p[stars[0]+1:stars[1]]
        C = p[stars[1]+1:]
        lenA, lenB, lenC = len(A), len(B), len(C)
        
        if A == "" and B == "" and C == "":
            return 0
        
        if B == "" and C == "":
            if A == "":
                return 0
            a_occurrences = [i for i in range(n - lenA + 1) if s[i:i+lenA] == A]
            return lenA if a_occurrences else -1
        
        if A == "":
            a_list = list(range(n + 1))
        else:
            a_list = [i for i in range(n - lenA + 1) if s[i:i+lenA] == A]
        
        if B == "":
            b_list = list(range(n + 1))
        else:
            b_list = [i for i in range(n - lenB + 1) if s[i:i+lenB] == B]
        
        inf_val = 10**15
        min_end_for_b = [inf_val] * len(b_list)
        
        if C != "":
            if C == "":
                c_list = list(range(n + 1))
            else:
                c_list = [i for i in range(n - lenC + 1) if s[i:i+lenC] == C]
            c_list.sort()
            min_end_arr = [inf_val] * len(c_list)
            if c_list:
                min_end_arr[-1] = c_list[-1] + lenC - 1
                for i in range(len(c_list)-2, -1, -1):
                    min_end_arr[i] = min(min_end_arr[i+1], c_list[i] + lenC - 1)
            for i, b in enumerate(b_list):
                req = b + lenB
                idx_c = bisect.bisect_left(c_list, req)
                if idx_c < len(c_list):
                    min_end_for_b[i] = min_end_arr[idx_c]
        else:
            for i, b in enumerate(b_list):
                min_end_for_b[i] = b + lenB - 1
        
        suff_min_end = [inf_val] * (len(b_list) + 1)
        for i in range(len(b_list)-1, -1, -1):
            suff_min_end[i] = min(min_end_for_b[i], suff_min_end[i+1])
        
        ans = inf_val
        for a in a_list:
            req_b_start = a + lenA
            idx_b = bisect.bisect_left(b_list, req_b_start)
            if idx_b < len(b_list):
                end_val = suff_min_end[idx_b]
                if end_val < inf_val:
                    length = end_val - a + 1
                    if length < ans:
                        ans = length
        return ans if ans < inf_val else -1