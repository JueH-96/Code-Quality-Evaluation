class Solution:
    def maximumLength(self, s: str) -> int:
        def count_overlapping(s, sub):
            count = 0
            l = len(sub)
            for i in range(len(s) - l + 1):
                if s[i:i + l] == sub:
                    count += 1
            return count
        
        for L in range(len(s), 0, -1):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                t = c * L
                cnt = count_overlapping(s, t)
                if cnt >= 3:
                    return L
        return -1