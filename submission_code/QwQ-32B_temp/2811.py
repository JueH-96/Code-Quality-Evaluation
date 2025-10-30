class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        forbidden = set()
        arr = []
        current = 1
        while len(arr) < n:
            if current not in forbidden:
                arr.append(current)
                forbidden.add(k - current)
            current += 1
        return sum(arr)