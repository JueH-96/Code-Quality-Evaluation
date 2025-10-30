class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        dp = {}
        for num in nums:
            new_dp = {}
            # Copy current dp to new_dp (not including current num)
            for key in dp:
                new_dp[key] = dp[key]
            # Process including the current num for each state in current dp
            for (last, c) in dp:
                current_length = dp[(last, c)]
                if last is None:
                    # Starting a new subsequence
                    new_last = num
                    new_c = 0
                    new_length = current_length + 1
                    if (new_last, new_c) in new_dp:
                        if new_length > new_dp[(new_last, new_c)]:
                            new_dp[(new_last, new_c)] = new_length
                    else:
                        new_dp[(new_last, new_c)] = new_length
                else:
                    if num == last:
                        new_last = num
                        new_c = c
                        new_length = current_length + 1
                        if (new_last, new_c) in new_dp:
                            if new_length > new_dp[(new_last, new_c)]:
                                new_dp[(new_last, new_c)] = new_length
                        else:
                            new_dp[(new_last, new_c)] = new_length
                    else:
                        if c < k:
                            new_last = num
                            new_c = c + 1
                            new_length = current_length + 1
                            if (new_last, new_c) in new_dp:
                                if new_length > new_dp[(new_last, new_c)]:
                                    new_dp[(new_last, new_c)] = new_length
                            else:
                                new_dp[(new_last, new_c)] = new_length
            # Add the possibility of starting a new subsequence with this num
            if (num, 0) in new_dp:
                if 1 > new_dp[(num, 0)]:
                    new_dp[(num, 0)] = 1
            else:
                new_dp[(num, 0)] = 1
            # Update dp to new_dp
            dp = new_dp.copy()
        if not dp:
            return 0
        max_len = max(dp.values())
        return max_len