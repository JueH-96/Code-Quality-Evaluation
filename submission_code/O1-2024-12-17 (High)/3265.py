class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        # We'll use a prefix sums array so we can quickly compute the sum of any subarray i..j
        # as prefixSums[j+1] - prefixSums[i].
        # A subarray is "good" if |nums[i] - nums[j]| == k, i.e. either nums[i] = nums[j]+k or nums[i] = nums[j]-k.
        #
        # Strategy:
        # 1. Create prefix sums array prefixSums, where prefixSums[i] = sum(nums[:i]).
        # 2. Keep a dictionary minPrefixForValue that maps a value (nums[i]) -> the smallest prefixSums[i]
        #    among indices i seen so far with that value.
        # 3. Iterate over each index j. Check if (nums[j] + k) in minPrefixForValue or (nums[j] - k) in minPrefixForValue.
        #    If yes, compute a candidate sum for a subarray ending at j (i..j) using the minimum prefix sum stored.
        # 4. Track the maximum sum found. If no such subarray exists, return 0.
        #
        # This works in O(n) time with dictionary lookups in average O(1).

        n = len(nums)
        prefixSums = [0] * (n + 1)
        for i in range(n):
            prefixSums[i+1] = prefixSums[i] + nums[i]

        minPrefixForValue = {}
        best = None  # We use None to detect if we've found any valid subarray at all

        for j in range(n):
            v = nums[j]

            # Check value v+k
            if (v + k) in minPrefixForValue:
                candidate_sum = prefixSums[j+1] - minPrefixForValue[v + k]
                if best is None or candidate_sum > best:
                    best = candidate_sum

            # Check value v-k
            if (v - k) in minPrefixForValue:
                candidate_sum = prefixSums[j+1] - minPrefixForValue[v - k]
                if best is None or candidate_sum > best:
                    best = candidate_sum

            # Update the dictionary with the current value v and prefixSums[j]
            # We store the minimum prefix sum among all indices having nums[i] = v
            if v not in minPrefixForValue:
                minPrefixForValue[v] = prefixSums[j]
            else:
                if prefixSums[j] < minPrefixForValue[v]:
                    minPrefixForValue[v] = prefixSums[j]

        return best if best is not None else 0