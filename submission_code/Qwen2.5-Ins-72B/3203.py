from collections import Counter

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half_n = n // 2
        s1 = s[:half_n]
        s2 = s[half_n:][::-1]
        
        # Precompute prefix counts for s1 and s2
        prefix_count1 = [Counter()]
        prefix_count2 = [Counter()]
        
        for i in range(half_n):
            prefix_count1.append(prefix_count1[-1] + Counter(s1[i]))
            prefix_count2.append(prefix_count2[-1] + Counter(s2[i]))
        
        def can_rearrange_to_match(s1, s2, a, b, c, d):
            # Check if the characters in s1[a:b+1] and s2[c:d+1] can be rearranged to match
            counter1 = prefix_count1[b + 1] - prefix_count1[a]
            counter2 = prefix_count2[d + 1] - prefix_count2[c]
            return counter1 == counter2
        
        def is_palindrome_possible(a, b, c, d):
            # Check if the characters outside the query ranges are already matching
            for i in range(a):
                if s1[i] != s2[i]:
                    return False
            for i in range(b + 1, c):
                if s1[i] != s2[i]:
                    return False
            for i in range(d + 1, half_n):
                if s1[i] != s2[i]:
                    return False
            
            # Check if the characters within the query ranges can be rearranged to match
            if not can_rearrange_to_match(s1, s2, a, b, c - half_n, d - half_n):
                return False
            
            return True
        
        return [is_palindrome_possible(a, b, c, d) for a, b, c, d in queries]