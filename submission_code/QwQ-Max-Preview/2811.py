class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        used = set()
        forbidden = set()
        current = 1
        total = 0
        for _ in range(n):
            while current in forbidden or current in used:
                current += 1
            total += current
            used.add(current)
            forbidden.add(k - current)
            current += 1
        return total