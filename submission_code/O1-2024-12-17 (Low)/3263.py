class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        """
        We are to split nums into exactly 3 disjoint contiguous subarrays that 
        cover the entire array. Each subarray's "cost" is its first element, 
        and we want the sum of those 3 first elements to be as small as possible.

        Because the 3 subarrays must cover the array in order, the first subarray
        must start at index 0 and go up to some index i, the second subarray must
        start at i+1 and go up to some index j, and the third subarray starts at
        j+1 until the end of the array. Hence, the total cost is:
            nums[0] + nums[i+1] + nums[j+1].

        We just need to try all valid pairs (i, j) such that:
            0 <= i < j < n - 1
            ensuring each subarray has at least one element.

        We'll keep track of the minimum sum of costs found.
        This yields an O(n^2) solution, which is efficient enough for n <= 50.
        """
        n = len(nums)
        ans = float('inf')
        
        # The first subarray is [0..i], second is [i+1..j], third is [j+1..n-1].
        # Thus cost = nums[0] + nums[i+1] + nums[j+1].
        # We need i < j, and j < n-1 to ensure each subarray has at least one element.
        for i in range(n - 2):        # i goes up to n-3 at most
            for j in range(i + 1, n - 1):
                cost = nums[0] + nums[i+1] + nums[j+1]
                ans = min(ans, cost)
        
        return ans