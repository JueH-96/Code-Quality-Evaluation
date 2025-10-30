class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def compute_xor_score(subarray):
            """Calculate the XOR score of a subarray according to the defined operations"""
            if len(subarray) == 1:
                return subarray[0]
            
            # Make a copy to avoid modifying the original array
            result = subarray.copy()
            
            # Apply the operations until only one element remains
            while len(result) > 1:
                new_result = []
                for i in range(len(result) - 1):
                    new_result.append(result[i] ^ result[i + 1])
                result = new_result
            
            return result[0]
        
        answer = []
        
        for l, r in queries:
            max_xor = 0  # Initialize to 0
            
            # Check all possible subarrays within the range [l, r]
            for i in range(l, r + 1):
                for j in range(i, r + 1):
                    subarray = nums[i:j + 1]
                    xor_score = compute_xor_score(subarray)
                    max_xor = max(max_xor, xor_score)
            
            answer.append(max_xor)
        
        return answer