from typing import List

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        mod = 10**9 + 7
        s_total = sum(nums)
        max_sum = s_total  # maximum possible sum
        # dp[s] will be the number of ways to get sum s using the processed numbers.
        dp = [0] * (max_sum + 1)
        dp[0] = 1

        # Count frequency of each distinct number.
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        # Process each distinct number.
        for num, count in freq.items():
            # Create new dp array for transition from dp to dp_new.
            dp_new = [0] * (max_sum + 1)
            # For each remainder mod num, we can do a grouped DP.
            # But a simpler method is to use prefix sums optimization.
            for s in range(0, max_sum + 1):
                dp_new[s] = dp[s]  # k = 0 case
            # To include multiples, we'll update dp_new using a sliding window approach.
            # For each starting position s, we want to add ways via picking 1,2,...,count of current number.
            # We can perform convolution type addition by iterating over residues modulo num.
            for remainder in range(num):
                # Build a list for indices in that residue class.
                arr = []
                indices = []
                index = remainder
                while index <= max_sum:
                    arr.append(dp[index])
                    indices.append(index)
                    index += num
                
                n = len(arr)
                # Use prefix sum and sliding window to compute convolution with polynomial 1 + x + ... + x^count.
                prefix = [0] * (n + 1)
                for i in range(n):
                    prefix[i+1] = (prefix[i] + arr[i]) % mod
                
                new_arr = [0] * n
                for i in range(n):
                    # sum from i-count to i inclusive if valid.
                    lo = max(0, i - count)
                    # The sum of dp values that contributed to index i is prefix[i+1] - prefix[lo]
                    new_arr[i] = (prefix[i+1] - prefix[lo]) % mod
                
                # Write back into dp_new.
                for idx, real_index in enumerate(indices):
                    dp_new[real_index] = new_arr[idx]
            dp = dp_new
        
        # Count solutions with sum in [l, r]
        result = 0
        # Bound r by max_sum.
        upper = min(r, max_sum)
        for s in range(l, upper + 1):
            result = (result + dp[s]) % mod
        return result