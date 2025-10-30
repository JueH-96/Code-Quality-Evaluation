class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        # Helper function to find maximum XOR of any subarray in nums[l..r]
        def maxXorInRange(l, r):
            max_xor = 0
            current_xor = 0
            # Use a set to store prefixes of XORs
            prefix_set = set()
            prefix_set.add(0)  # Initialize with 0 to handle single element subarrays

            for i in range(l, r + 1):
                current_xor ^= nums[i]
                # Try to maximize the XOR with the current prefix
                for prefix in prefix_set:
                    max_xor = max(max_xor, current_xor ^ prefix)
                # Add current prefix to the set
                prefix_set.add(current_xor)

            return max_xor

        result = []
        for l, r in queries:
            result.append(maxXorInRange(l, r))
        
        return result