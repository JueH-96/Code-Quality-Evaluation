class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        forbidden = set()
        array = []
        current = 1
        while len(array) < n:
            if current not in forbidden:
                array.append(current)
                forbidden.add(k - current)
            current += 1
        return sum(array)