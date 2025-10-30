class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        current_sum = 0
        num_set = set()
        num = 1
        added = 0
        while added < n:
            if (k - num) not in num_set:
                num_set.add(num)
                current_sum += num
                added += 1
            num += 1
        return current_sum