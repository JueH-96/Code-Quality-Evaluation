class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        res = []
        for q in queries:
            a, b, c, d = q
            valid = True
            
            # Step 1: Check non-query regions
            for i in range(n // 2):
                j = n - 1 - i
                if (i < a or i > b) and (j < c or j > d):
                    if s[i] != s[j]:
                        valid = False
                        break
            if not valid:
                res.append(False)
                continue
            
            # Step 2: Check a..b's i where j not in c..d
            first_sub = s[a:b+1]
            set_first = set(first_sub)
            for i in range(a, b + 1):
                j = n - 1 - i
                if not (c <= j <= d):
                    if s[j] not in set_first:
                        valid = False
                        break
            if not valid:
                res.append(False)
                continue
            
            # Step 3: Check c..d's j where i not in a..b
            second_sub = s[c:d+1]
            set_second = set(second_sub)
            for j in range(c, d + 1):
                i = n - 1 - j
                if not (a <= i <= b):
                    if s[i] not in set_second:
                        valid = False
                        break
            if not valid:
                res.append(False)
                continue
            
            # Step 4: Check for overlap and common characters
            min_j = n - 1 - b
            max_j = n - 1 - a
            has_overlap = (min_j <= d) and (max_j >= c)
            if has_overlap:
                common = set_first & set_second
                if not common:
                    valid = False
            
            res.append(valid)
        return res