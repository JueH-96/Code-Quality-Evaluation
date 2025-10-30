class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        t = []
        k_remaining = k
        for c in s:
            for target in 'abcdefghijklmnopqrstuvwxyz':
                distance = abs(ord(c) - ord(target))
                distance = min(distance, 26 - distance)
                if distance <= k_remaining:
                    t.append(target)
                    k_remaining -= distance
                    break
        return ''.join(t)