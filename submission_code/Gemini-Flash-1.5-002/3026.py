class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        nums = []
        total_sum = 0
        used = set()

        for i in range(1, target // 2 + 2):
            if i not in used and target - i not in used:
                nums.append(i)
                total_sum = (total_sum + i) % MOD
                used.add(i)
                if len(nums) == n:
                    break
                if target -i > i and target -i not in used:
                    nums.append(target - i)
                    total_sum = (total_sum + target - i) % MOD
                    used.add(target -i)
                    if len(nums) == n:
                        break

        if len(nums) < n:
            next_val = 1
            while len(nums) < n:
                if next_val not in used:
                    nums.append(next_val)
                    total_sum = (total_sum + next_val) % MOD
                    used.add(next_val)
                next_val += 1

        return total_sum