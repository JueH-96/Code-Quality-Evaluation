class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        total_subarrays = n * (n + 1) // 2

        def count_subarrays_with_at_most_k_distinct(k):
            count = 0
            for i in range(n):
                distinct_count = 0
                seen = set()
                for j in range(i, n):
                    if nums[j] not in seen:
                        distinct_count += 1
                        seen.add(nums[j])
                    if distinct_count <= k:
                        count += 1
                    else:
                        break
            return count

        distinct_values = sorted(list(set(nums)))
        low = 1
        high = len(distinct_values)
        median_value = high

        while low <= high:
            mid = (low + high) // 2
            count = count_subarrays_with_at_most_k_distinct(mid)

            if count * 2 < total_subarrays + (total_subarrays % 2):
                low = mid + 1
            else:
                median_value = mid
                high = mid - 1

        return median_value