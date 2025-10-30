class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        selected = set()
        forbidden = set()
        current = 1
        while len(selected) < n:
            if current not in selected and current not in forbidden:
                selected.add(current)
                complement = k - current
                if complement > 0:
                    forbidden.add(complement)
            current += 1
        return sum(selected)