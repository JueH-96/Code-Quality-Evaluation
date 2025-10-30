class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        selected = set()
        total_sum = 0
        current = 1
        while len(selected) < n:
            if (k - current) not in selected:
                selected.add(current)
                total_sum += current
            current += 1
        return total_sum