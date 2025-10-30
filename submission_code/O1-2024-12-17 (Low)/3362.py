class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        # We want the "median" among all subarray distinct counts.
        # Let total_sub = number of subarrays = n*(n+1)//2.
        # The median index M = (total_sub + 1) // 2 (1-based).
        # We'll use a sliding-window count for "at most k distinct" elements
        # and do a binary search over k in [1..n].

        # subarr_count(k) = number of subarrays with distinctness <= k.
        # Then we find the smallest k where subarr_count(k) >= M.

        def subarrays_with_at_most_k_distinct(arr, k):
            # Counts how many subarrays in arr have at most k distinct elements
            count = 0
            left = 0
            freq = {}
            distinct_count = 0

            for right, val in enumerate(arr):
                freq[val] = freq.get(val, 0) + 1
                if freq[val] == 1:
                    distinct_count += 1
                while distinct_count > k:
                    freq[arr[left]] -= 1
                    if freq[arr[left]] == 0:
                        distinct_count -= 1
                    left += 1
                # All subarrays that end at right and have at most k distinct
                # start anywhere from left to right
                count += (right - left + 1)
            return count

        n = len(nums)
        total_sub = n * (n + 1) // 2
        median_index = (total_sub + 1) // 2  # 1-based index of the median

        # Binary search for smallest k where subarr_count(k) >= median_index
        left, right = 1, n
        ans = 1
        while left <= right:
            mid = (left + right) // 2
            if subarrays_with_at_most_k_distinct(nums, mid) >= median_index:
                ans = mid
                right = mid - 1
            else:
                left = mid + 1

        return ans