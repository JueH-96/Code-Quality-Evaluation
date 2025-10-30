class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        n = len(nums)
        max_product = -1
        
        # Try all possible subsequences using bitmask
        for mask in range(1, 1 << n):
            subsequence = []
            indices = []
            
            # Build subsequence based on bitmask
            for i in range(n):
                if mask & (1 << i):
                    subsequence.append(nums[i])
                    indices.append(i)
            
            # Calculate alternating sum
            alt_sum = 0
            for j in range(len(subsequence)):
                if j % 2 == 0:
                    alt_sum += subsequence[j]
                else:
                    alt_sum -= subsequence[j]
            
            # Check if alternating sum equals k
            if alt_sum == k:
                # Calculate product
                product = 1
                for num in subsequence:
                    product *= num
                
                # Update max_product if within limit
                if product <= limit:
                    max_product = max(max_product, product)
        
        return max_product