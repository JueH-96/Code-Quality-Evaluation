import bisect

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # Find all starting indices of 'a' in 's'
        len_a = len(a)
        A = []
        for i in range(len(s) - len_a + 1):
            if s[i:i+len_a] == a:
                A.append(i)
        
        # Find all starting indices of 'b' in 's'
        len_b = len(b)
        B = []
        for i in range(len(s) - len_b + 1):
            if s[i:i+len_b] == b:
                B.append(i)
        
        result = []
        for i in A:
            low = i - k
            high = i + k
            # Find the first position in B where the value is >= low
            pos = bisect.bisect_left(B, low)
            if pos < len(B) and B[pos] <= high:
                result.append(i)
        
        return result