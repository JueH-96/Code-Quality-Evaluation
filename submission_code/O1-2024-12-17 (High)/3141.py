from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        """
        We want the length of the shortest subarray in the infinite repetition
        of nums that sums to target, or -1 if no such subarray exists.

        Key ideas (all nums[i] > 0):
        1) Let S = sum(nums).
        2) If S == target, the entire array (length = len(nums)) is the only
           sub-subarray summing to S (since all positives), so answer = len(nums).
        3) If S > target, it suffices to search for a sub-subarray of sum = target
           in at most 2 copies of nums (sliding-window). If none found, answer = -1.
        4) If S < target:
           - Let k = target // S and r = target % S. Then target = k*S + r.
           - If r == 0, we can form target by taking exactly k copies of nums in a row;
             that subarray has length = k * len(nums), which must be minimal for all-positives.
           - Otherwise, we use two approaches:
             A) Take k full copies (sum = k*S) plus the minimal sub-subarray that sums to r
                within 2 copies of nums.  Total length = k*len(nums) + that minimal length.
             B) Take (k+1) full copies (sum = (k+1)*S) and skip a sub-subarray of sum = (S - r).
                This leaves exactly k*S + r.  The length of what we keep is
                    (k+1)*len(nums) - (length of that sub-subarray),
                so we want the largest sub-subarray summing to (S - r) to minimize the kept length.
             We return the min of those two lengths (if both ways find a match). Otherwise -1.
        """

        n = len(nums)
        S = sum(nums)

        # Quick check #1
        if S == target:
            return n

        # Helper: find the minimal-length sub-subarray summing to x in nums+nums
        def find_min_subarray_sum_x_in_2copies(arr: List[int], x: int) -> int:
            """
            Returns the minimum length of a sub-subarray in (arr+arr)
            that sums exactly to x, or -1 if none exists.
            """
            total = sum(arr)
            if x > 2 * total:
                return -1  # can't form x if x exceeds sum of 2 copies

            combined = arr + arr
            start = 0
            cur_sum = 0
            min_len = float('inf')
            for end in range(len(combined)):
                cur_sum += combined[end]
                while cur_sum >= x:
                    if cur_sum == x:
                        min_len = min(min_len, end - start + 1)
                    # shrink from the left
                    cur_sum -= combined[start]
                    start += 1
            return min_len if min_len != float('inf') else -1

        # Helper: find the maximal-length sub-subarray summing to x in nums+nums
        def find_max_subarray_sum_x_in_2copies(arr: List[int], x: int) -> int:
            """
            Returns the maximum length of a sub-subarray in (arr+arr)
            that sums exactly to x, or -1 if none exists.
            """
            total = sum(arr)
            if x > 2 * total:
                return -1

            combined = arr + arr
            start = 0
            cur_sum = 0
            max_len = -1
            for end in range(len(combined)):
                cur_sum += combined[end]
                while cur_sum > x:
                    cur_sum -= combined[start]
                    start += 1
                if cur_sum == x:
                    max_len = max(max_len, end - start + 1)
            return max_len

        # Case 2: S > target => just try to find sub-subarray of sum=target in up to 2 copies
        if S > target:
            ans = find_min_subarray_sum_x_in_2copies(nums, target)
            return ans

        # Case 3: S < target
        k = target // S
        r = target % S

        # If target is multiple of S => entire k copies is the only way
        if r == 0:
            # k copies each of sum S => total length = k*n
            return k * n

        # Otherwise, we check approach A and approach B.
        # A) k copies + minimal sub-subarray = r
        # B) (k+1) copies - largest sub-subarray = (S-r)

        # Approach A: minimal sub-subarray summing to r
        min_r_len = find_min_subarray_sum_x_in_2copies(nums, r)
        ansA = (k * n + min_r_len) if min_r_len != -1 else float('inf')

        # Approach B: skip sub-subarray with sum = (S - r) from (k+1) copies
        # so total subarray has sum = target
        if S - r == 0:
            # that means r = S => subarray sum needed is entire (k+1)*S,
            # skipping sub-subarray of sum=0 can't happen with positives => skip is length=0
            # so subarray length = (k+1)*n
            ansB = (k + 1) * n
        else:
            max_sr_len = find_max_subarray_sum_x_in_2copies(nums, S - r)
            ansB = (k + 1) * n - max_sr_len if max_sr_len != -1 else float('inf')

        ans = min(ansA, ansB)
        return ans if ans != float('inf') else -1