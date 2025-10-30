from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        n = len(nums)

        # Calculate the initial frequency of k
        total_k_freq = 0
        for num in nums:
            if num == k:
                total_k_freq += 1

        # Base case: no operation or x=0. Max frequency is the initial frequency of k.
        max_freq = total_k_freq

        # Iterate through all possible values v (1 to 50) that elements inside the subarray
        # could initially have and be changed to k.
        # This corresponds to choosing x = k - v.
        # We only need to consider v values that exist in the original nums, because if v is not in nums,
        # Freq(v in i..j) will be 0 for any i, j, making the operation useless for changing values inside.
        # The range of nums[i] is [1, 50], so we check v from 1 to 50.
        for v in range(1, 51):
            # If v = k, x = k - k = 0. The operation adds 0 to the subarray, changing nothing.
            # The frequency of k remains TotalFreq(k). This case is handled by the initial max_freq.
            if v == k:
                continue

            # For a fixed v != k, we apply x = k - v to a subarray nums[i..j].
            # The final frequency of k will be:
            # count(m | i <= m <= j and nums[m] == v) + count(m | m < i or m > j and nums[m] == k)
            # = count(m | i <= m <= j and nums[m] == v) + (TotalFreq(k) - count(m | i <= m <= j and nums[m] == k))
            # = (count(m | i <= m <= j and nums[m] == v) - count(m | i <= m <= j and nums[m] == k)) + TotalFreq(k)
            # We want to maximize the term (count(m | i <= m <= j and nums[m] == v) - count(m | i <= m <= j and nums[m] == k))
            # over all 0 <= i <= j < n.
            # Let c[m] = 1 if nums[m] == v, -1 if nums[m] == k, 0 otherwise.
            # We find the maximum subarray sum of this c array (must be non-empty subarray).

            # Kadane's algorithm for maximum non-empty subarray sum (can be negative)
            # Initialize max_diff and current_max_diff based on the first element.
            # Check for empty array case, though constraints say n >= 1.
            if n == 0:
                 # This case is not reachable based on constraints n >= 1
                 continue

            c_val_0 = 0
            if nums[0] == v:
                c_val_0 = 1
            elif nums[0] == k:
                c_val_0 = -1
            
            max_diff = c_val_0
            current_max_diff = c_val_0

            for m in range(1, n):
                c_val_m = 0
                if nums[m] == v:
                    c_val_m = 1
                elif nums[m] == k:
                    c_val_m = -1

                # The max sum ending at index m is either just c_val_m itself
                # (starting a new subarray from m)
                # or the max sum ending at m-1 plus c_val_m (extending the previous subarray).
                current_max_diff = max(c_val_m, current_max_diff + c_val_m)

                # The overall max sum is the maximum of max sums ending at each index up to m.
                max_diff = max(max_diff, current_max_diff)

            # The maximum frequency for this chosen v is max_diff + total_k_freq
            # max_diff represents the maximum increase in count (Freq(v in i..j) - Freq(k in i..j))
            # achieved by choosing the optimal non-empty subarray for this v.
            # This increase can be negative. Adding it to total_k_freq gives the resulting frequency.
            max_freq = max(max_freq, max_diff + total_k_freq)

        return max_freq