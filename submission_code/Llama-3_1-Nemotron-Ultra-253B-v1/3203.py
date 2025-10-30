from typing import List

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half = n // 2
        
        # Precompute bad array and its prefix sum
        prefix_bad = [0] * (half + 1)
        for i in range(half):
            j = n - 1 - i
            prefix_bad[i+1] = prefix_bad[i] + (s[i] != s[j])
        
        # Precompute prefix sums for each character in the first half
        pre_first = [[0]*(half + 1) for _ in range(26)]
        for i in range(half):
            c = ord(s[i]) - ord('a')
            for x in range(26):
                pre_first[x][i+1] = pre_first[x][i] + (x == c)
        
        # Precompute prefix sums for each character in the second half
        pre_second = [[0]*(half + 1) for _ in range(26)]
        for i in range(half, n):
            c = ord(s[i]) - ord('a')
            idx = i - half
            for x in range(26):
                pre_second[x][idx+1] = pre_second[x][idx] + (x == c)
        
        res = []
        for q in queries:
            a, b, c, d = q
            len_first = b - a + 1
            len_second = d - c + 1
            if (len_first + len_second) % 2 != 0:
                res.append(False)
                continue
            
            # Calculate sum of non-covered bad pairs
            total_bad = prefix_bad[-1]
            sum_a_b = prefix_bad[b+1] - prefix_bad[a]
            j_start = n - 1 - d
            j_end = n - 1 - c
            sum_j_range = prefix_bad[j_end + 1] - prefix_bad[j_start]
            
            intersection_start = max(a, j_start)
            intersection_end = min(b, j_end)
            sum_intersection = 0
            if intersection_start <= intersection_end:
                sum_intersection = prefix_bad[intersection_end + 1] - prefix_bad[intersection_start]
            
            sum_non_covered = total_bad - sum_a_b - sum_j_range + sum_intersection
            if sum_non_covered > 0:
                res.append(False)
                continue
            
            # Check combined multiset of allowed ranges
            # First allowed range [a, b] in first half
            count_first = [0] * 26
            for x in range(26):
                count_first[x] = pre_first[x][b+1] - pre_first[x][a]
            
            # Second allowed range [c, d] in second half, convert to 0-based index in second half
            second_start = c - half
            second_end = d - half
            count_second = [0] * 26
            for x in range(26):
                count_second[x] = pre_second[x][second_end + 1] - pre_second[x][second_start]
            
            valid = True
            for x in range(26):
                total = count_first[x] + count_second[x]
                if total % 2 != 0:
                    valid = False
                    break
            res.append(valid)
        
        return res