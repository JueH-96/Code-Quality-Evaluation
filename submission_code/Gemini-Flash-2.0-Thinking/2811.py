class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        arr = []
        seen = set()
        num = 1
        while len(arr) < n:
            if k - num not in seen:
                arr.append(num)
                seen.add(num)
            num += 1
        return sum(arr)