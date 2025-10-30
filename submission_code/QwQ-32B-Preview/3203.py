from typing import List
from collections import defaultdict

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        mid = n // 2
        s_reversed = s[::-1]
        answer = []
        
        for query in queries:
            a, b, c, d = query
            # Adjust c to be relative to the second half
            c += mid
            d += mid
            
            # Fixed parts
            first_fixed = s[0:a] + s[b+1:mid]
            second_fixed = s[mid:c] + s[d+1:n]
            
            # Rearrangeable parts
            first_rearr = s[a:b+1]
            second_rearr = s[c:d+1]
            
            # Frequency counts for rearrangeable parts
            freq1 = defaultdict(int)
            for char in first_rearr:
                freq1[char] += 1
            freq2 = defaultdict(int)
            for char in second_rearr:
                freq2[char] += 1
            
            # Check fixed parts
            possible = True
            i = 0
            while i < len(first_fixed):
                if first_fixed[i] != s_reversed[mid - i - 1]:
                    # Mismatch in fixed parts
                    # Try to fix using rearrangeable parts
                    char_needed = s_reversed[mid - i - 1]
                    if a <= a + i <= b and c <= d - i <= d:
                        # Both positions are in rearrangeable parts
                        if freq1.get(char_needed, 0) > 0:
                            freq1[char_needed] -= 1
                        else:
                            possible = False
                            break
                    elif a <= a + i <= b:
                        # Only i is in first rearrangeable
                        if freq1.get(char_needed, 0) > 0:
                            freq1[char_needed] -= 1
                        else:
                            possible = False
                            break
                    elif c <= d - i <= d:
                        # Only j is in second rearrangeable
                        if freq2.get(char_needed, 0) > 0:
                            freq2[char_needed] -= 1
                        else:
                            possible = False
                            break
                    else:
                        # Both are fixed and don't match
                        possible = False
                        break
                i += 1
            
            # Check if frequency counts are equal after fixes
            if possible:
                if freq1 == freq2:
                    answer.append(True)
                else:
                    answer.append(False)
            else:
                answer.append(False)
        
        return answer