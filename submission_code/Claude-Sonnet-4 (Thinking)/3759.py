class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        
        # Get unique values in nums1
        unique_vals = sorted(set(nums1))
        
        # For each unique value, precompute the top k sum
        precomputed = {}
        for val in unique_vals:
            candidates = [nums2[j] for j in range(n) if nums1[j] < val]
            candidates.sort(reverse=True)
            precomputed[val] = sum(candidates[:k])
        
        # Build answer array
        return [precomputed[nums1[i]] for i in range(n)]