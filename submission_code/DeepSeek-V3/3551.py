class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        # Precompute the prefix XOR array
        prefix_xor = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            prefix_xor[i+1] = prefix_xor[i] ^ nums[i]
        
        # Function to compute the XOR score of a subarray
        def get_xor_score(l, r):
            # The XOR score of a subarray nums[l..r] is the XOR of all elements in the subarray
            return prefix_xor[r+1] ^ prefix_xor[l]
        
        # For each query, find the maximum XOR score of any subarray within the range
        answer = []
        for l, r in queries:
            max_score = 0
            # Iterate over all possible subarrays within [l, r]
            for i in range(l, r+1):
                current_xor = 0
                for j in range(i, r+1):
                    current_xor ^= nums[j]
                    if current_xor > max_score:
                        max_score = current_xor
            answer.append(max_score)
        return answer