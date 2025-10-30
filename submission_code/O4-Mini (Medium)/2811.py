class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        chosen = set()
        res = 0
        i = 1
        count = 0
        while count < n:
            # we can pick i if k - i is not already chosen
            if (k - i) not in chosen:
                chosen.add(i)
                res += i
                count += 1
            i += 1
        return res