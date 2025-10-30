class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        if k <= 1:
            return n * (n + 1) // 2
        avoid = set()
        total = 0
        i = 1
        while len(avoid) < n:
            if k - i not in avoid:
                avoid.add(i)
                total += i
            i += 1
        return total