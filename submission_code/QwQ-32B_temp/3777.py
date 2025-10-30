class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        from collections import defaultdict

        # Initialize DP: key is (sum, parity), value is the maximum product
        dp = {(0, 0): 1}  # Starting with empty subsequence (sum 0, parity 0, product 1)

        for num in nums:
            temp_dp = dp.copy()  # Start with not taking the current number

            # Iterate over the current DP states to consider taking the current number
            for (current_sum, parity), current_prod in dp.items():
                contribution = num if (parity == 0) else -num
                new_sum = current_sum + contribution
                new_parity = (parity + 1) % 2
                new_prod = current_prod * num

                if new_prod > limit:
                    continue  # Skip if product exceeds the limit

                key = (new_sum, new_parity)
                # Update the temp_dp if this new product is better
                if key in temp_dp:
                    if new_prod > temp_dp[key]:
                        temp_dp[key] = new_prod
                else:
                    temp_dp[key] = new_prod

            dp = temp_dp  # Update DP to the new state after considering this number

        max_prod = -1
        for (s, p), prod in dp.items():
            if s == k and prod != 1:  # Exclude the empty subsequence (product 1)
                if prod > max_prod:
                    max_prod = prod

        return max_prod if max_prod != -1 else -1