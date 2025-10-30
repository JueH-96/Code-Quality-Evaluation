class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        star_idx = p.index('*')
        left = p[:star_idx]
        right = p[star_idx + 1:]
        len_left = len(left)
        len_right = len(right)
        
        if len_left > 0:
            pos = 0
            while True:
                idx = s.find(left, pos)
                if idx == -1:
                    break
                if s.find(right, idx + len_left) != -1:
                    return True
                pos = idx + 1
            return False
        else:  # len_left == 0
            if len_right > 0:
                return s.find(right) != -1
            else:
                return True