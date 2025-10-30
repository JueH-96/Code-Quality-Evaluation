class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        forbidden = set()
        arr = []
        x = 1
        while len(arr) < n:
            if x not in forbidden:
                arr.append(x)
                forbidden.add(k - x)
            x += 1
        return sum(arr)