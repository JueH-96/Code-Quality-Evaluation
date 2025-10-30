import bisect

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        n = len(s)
        len_a = len(a)
        len_b = len(b)
        
        a_occurrences = []
        for i in range(n - len_a + 1):
            if s.startswith(a, i):
                a_occurrences.append(i)
                
        b_occurrences = []
        for i in range(n - len_b + 1):
            if s.startswith(b, i):
                b_occurrences.append(i)
        
        if not b_occurrences:
            return []
        
        b_occurrences.sort()
        
        result = []
        for i in a_occurrences:
            left_bound = i - k
            right_bound = i + k
            idx = bisect.bisect_left(b_occurrences, left_bound)
            if idx < len(b_occurrences) and b_occurrences[idx] <= right_bound:
                result.append(i)
        
        return result