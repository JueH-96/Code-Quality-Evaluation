class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        left, right = 1, n
        while left < right:
            mid = (left + right) // 2
            alice, bob = 0, 0
            for i in range(mid, n, 2):
                if possible[i]:
                    alice += 1
                else:
                    alice += 1
                    bob += 1
            for i in range(1, mid, 2):
                if possible[i]:
                    bob += 1
            if alice > bob:
                right = mid
            else:
                left = mid + 1
        return left