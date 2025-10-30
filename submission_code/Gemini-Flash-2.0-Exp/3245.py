class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        beautiful = []
        a_indices = []
        b_indices = []
        for i in range(len(s) - len(a) + 1):
            if s[i:i + len(a)] == a:
                a_indices.append(i)
        for i in range(len(s) - len(b) + 1):
            if s[i:i + len(b)] == b:
                b_indices.append(i)
        
        for i in a_indices:
            is_beautiful = False
            for j in b_indices:
                if abs(j - i) <= k:
                    is_beautiful = True
                    break
            if is_beautiful:
                beautiful.append(i)
        
        return sorted(beautiful)