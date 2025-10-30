class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        
        # Precompute XOR prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] ^ nums[i]
        
        # Function to calculate XOR score of a subarray
        def xor_score(left, right):
            score = 0
            for i in range(31, -1, -1):
                mask = 1 << i
                if (prefix[right + 1] ^ prefix[left]) & mask:
                    score |= mask
            return score
        
        # Process queries
        answer = []
        for l, r in queries:
            max_score = 0
            for i in range(l, r + 1):
                for j in range(i, r + 1):
                    score = xor_score(i, j)
                    max_score = max(max_score, score)
            answer.append(max_score)
        
        return answer