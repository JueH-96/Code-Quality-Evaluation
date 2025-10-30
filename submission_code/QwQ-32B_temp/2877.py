from itertools import permutations

class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(x, y):
            if x in y:
                return y
            if y in x:
                return x
            max_overlap_xy = 0
            len_x, len_y = len(x), len(y)
            # Find maximum overlap of x's suffix with y's prefix
            for k in range(1, min(len_x, len_y) + 1):
                if x.endswith(y[:k]):
                    max_overlap_xy = k
            option1 = x + y[max_overlap_xy:]
            
            max_overlap_yx = 0
            # Find maximum overlap of y's suffix with x's prefix
            for k in range(1, min(len_y, len_x) + 1):
                if y.endswith(x[:k]):
                    max_overlap_yx = k
            option2 = y + x[max_overlap_yx:]
            
            if len(option1) < len(option2):
                return option1
            elif len(option2) < len(option1):
                return option2
            else:
                return min(option1, option2)
        
        candidates = []
        for perm in permutations([a, b, c]):
            x, y, z = perm
            temp = merge(x, y)
            merged = merge(temp, z)
            candidates.append(merged)
        
        # Sort candidates by length, then lexicographical order
        candidates.sort(key=lambda s: (len(s), s))
        return candidates[0]