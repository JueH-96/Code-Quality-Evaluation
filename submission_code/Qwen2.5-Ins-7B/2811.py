class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        if k <= 2:
            return n * (n + 1) // 2
        avoid_set = set()
        result = 0
        for i in range(1, k // 2 + 1):
            avoid_set.add(k - i)
        count = 0
        i = 1
        while count < n:
            if i not in avoid_set:
                result += i
                count += 1
            i += 1
        return result