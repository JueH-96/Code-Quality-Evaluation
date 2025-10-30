class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        from collections import defaultdict
        
        # Prefix sum array to count the occurrences of remainders
        prefix_sum = [0]
        for num in nums:
            prefix_sum.append((prefix_sum[-1] + (num % modulo == k)) % modulo)
        
        # Dictionary to store the frequency of prefix sums
        prefix_count = defaultdict(int)
        interesting_count = 0
        
        # Iterate over the prefix sum array
        for sum_val in prefix_sum:
            # If the current prefix sum satisfies the condition, increment the count
            interesting_count += prefix_count[sum_val]
            # Increment the frequency of the current prefix sum
            prefix_count[sum_val] += 1
        
        return interesting_count