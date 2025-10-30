class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        from collections import defaultdict

        N = len(nums)
        total_subarrays = N * (N + 1) // 2

        K_max = 1000  # Adjust K_max as needed

        cumulative_count = 0
        counts = [0] * (K_max + 2)  # 1-indexed array

        # Function to compute number of subarrays with at most K distinct elements
        def subarrays_at_most_K(nums, K):
            count = 0
            freq = defaultdict(int)
            left = 0
            for right in range(N):
                freq[nums[right]] += 1
                while len(freq) > K:
                    freq[nums[left]] -=1
                    if freq[nums[left]] == 0:
                        del freq[nums[left]]
                    left += 1
                count += right - left +1
            return count

        previous_total = 0

        for K in range(1, K_max + 1):
            total_with_K = subarrays_at_most_K(nums, K)
            counts[K] = total_with_K - previous_total
            previous_total = total_with_K
            cumulative_count += counts[K]

            if cumulative_count >= (total_subarrays + 1) // 2:
                # Found median
                return K

        # If we did not find median within K_max, set K to maximum possible distinct elements
        # Compute counts for K greater than K_max
        # Compute number of distinct elements in nums
        distinct_elements = len(set(nums))
        for K in range(K_max + 1, distinct_elements + 1):
            total_with_K = subarrays_at_most_K(nums, K)
            counts[K] = total_with_K - previous_total
            previous_total = total_with_K
            cumulative_count += counts[K]
            if cumulative_count >= (total_subarrays + 1) // 2:
                return K

        # If still not found, return the maximum possible K
        return distinct_elements