class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        from collections import defaultdict

        count = defaultdict(int)
        zeros = 0
        for num in nums:
            if num == 0:
                zeros += 1
            else:
                count[num] += 1

        elements = list(count.keys())
        sqrt_r = int(r**0.5)

        # Initialize DP array
        dp = [0] * (r + 1)
        dp[0] = 1
        current_max = 0

        for x in elements:
            cnt = count[x]
            if x == 0:
                continue

            if x <= sqrt_r:
                # Process using sliding window for small x
                temp = dp.copy()
                current_sum = 0
                for s in range(x, r + 1, x):
                    prev_s = s - x
                    if prev_s >= 0:
                        current_sum += dp[prev_s]
                    # Subtract terms beyond cnt
                    if s > x * cnt:
                        subtract_s = s - (cnt + 1) * x
                        if subtract_s >= 0:
                            current_sum -= dp[subtract_s]
                    temp[s] = (temp[s] + current_sum) % MOD
                dp = temp
                current_max = min(current_max + cnt * x, r)
            else:
                # Process using standard method for large x
                temp = dp.copy()
                for s in range(current_max + 1):
                    if dp[s] == 0:
                        continue
                    val = dp[s]
                    for k in range(1, cnt + 1):
                        new_s = s + k * x
                        if new_s > r:
                            break
                        temp[new_s] = (temp[new_s] + val) % MOD
                dp = temp
                current_max = min(current_max + cnt * x, r)

        # Calculate the total valid subsets
        total = 0
        for s in range(l, r + 1):
            total = (total + dp[s]) % MOD

        # Multiply by (zeros + 1) to account for zero elements
        total = (total * (zeros + 1)) % MOD
        return total