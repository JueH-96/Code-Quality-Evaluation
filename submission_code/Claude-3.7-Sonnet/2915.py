class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        # Convert array: 1 if element satisfies condition, 0 otherwise
        for i in range(len(nums)):
            nums[i] = 1 if nums[i] % modulo == k else 0
        
        # Count interesting subarrays using prefix sums
        prefix_count = 0
        count_map = {0: 1}  # Initialize with 0 having frequency 1
        interesting_count = 0
        
        for num in nums:
            # Update prefix count
            prefix_count += num
            
            # Calculate the target remainder we need to find
            target = (prefix_count % modulo - k + modulo) % modulo
            
            # Add count of previous prefix sums that would make current subarray interesting
            interesting_count += count_map.get(target, 0)
            
            # Update frequency map
            count_map[prefix_count % modulo] = count_map.get(prefix_count % modulo, 0) + 1
        
        return interesting_count