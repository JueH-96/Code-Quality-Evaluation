class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        arr = []
        used = set()
        i = 1
        while len(arr) < n:
            if i not in used:
                arr.append(i)
            used.add(k-i)
            i += 1
        return sum(arr)