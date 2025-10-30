class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        # We will sum up (#distinct_elements_in_subarray) and (#consecutive_pairs_in_subarray)
        # for all subarrays, then use the formula:
        #   total_imbalance = sum_distinct - total_subarrays - sum_consecutive_pairs
        # where total_subarrays = n*(n+1)//2.

        total_subarrays = n * (n + 1) // 2
        sum_distinct = 0
        sum_pairs = 0
        
        for start in range(n):
            freq = [0] * (n + 1)          # frequency array for values [1..n]
            present = [False] * (n + 1)   # track whether a value is present in subarray
            distinct = 0
            pairs = 0
            for end in range(start, n):
                x = nums[end]
                freq[x] += 1
                if freq[x] == 1:
                    # x was not present before, so it's a new distinct value
                    present[x] = True
                    distinct += 1
                    # check consecutive neighbors
                    if x > 1 and present[x - 1]:
                        pairs += 1
                    if x < n and present[x + 1]:
                        pairs += 1
                # accumulate counts
                sum_distinct += distinct
                sum_pairs += pairs
        
        # compute final answer
        return sum_distinct - total_subarrays - sum_pairs