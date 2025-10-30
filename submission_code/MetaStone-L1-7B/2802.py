class Solution:
    def punishmentNumber(self, n: int) -> int:
        def is_valid(s_str, target):
            n = len(s_str)
            dp = [set() for _ in range(n + 1)]
            dp[0].add(0)
            for pos in range(n):
                current_sums = dp[pos]
                for sum_val in current_sums:
                    if sum_val > target:
                        continue
                    remaining = target - sum_val
                    if remaining <= 0:
                        continue
                    max_l = n - pos
                    for l in range(1, max_l + 1):
                        substring = s_str[pos:pos + l]
                        v = int(substring)
                        new_sum = sum_val + v
                        if new_sum > target:
                            continue
                        next_pos = pos + l
                        if next_pos == n and new_sum == target:
                            return True
                        if v > remaining:
                            break
                        dp[next_pos].add(new_sum)
            return False

        total = 0
        for i in range(1, n + 1):
            s = i * i
            s_str = str(s)
            if is_valid(s_str, i):
                total += s
        return total