class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        from itertools import combinations
        
        def alternating_sum_and_product(subseq):
            alt_sum = sum(subseq[i] if i % 2 == 0 else -subseq[i] for i in range(len(subseq)))
            product = 1
            for num in subseq:
                product *= num
                if product > limit:  # Early stop if product exceeds limit
                    return None, None
            return alt_sum, product
        
        max_product = -1
        
        # Check all possible subsequences
        for length in range(1, len(nums) + 1):
            for subseq in combinations(nums, length):
                alt_sum, product = alternating_sum_and_product(subseq)
                if alt_sum == k and product <= limit:
                    max_product = max(max_product, product)
        
        return max_product