class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        chosen = set()
        total = 0
        i = 1
        count = 0
        # Greedily pick the smallest positive integers,
        # but skip any i whose complement k-i is already chosen
        while count < n:
            if (k - i) not in chosen:
                chosen.add(i)
                total += i
                count += 1
            i += 1
        return total