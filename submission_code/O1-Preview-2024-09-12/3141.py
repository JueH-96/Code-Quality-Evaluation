class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        S = sum(nums)
        residual = target % S
        N = len(nums)
        if residual == 0:
            residual = S
        k = (target - residual) // S
        nums_doubled = nums + nums
        min_length = float('inf')
        left, curr_sum = 0, 0
        for right in range(len(nums_doubled)):
            curr_sum += nums_doubled[right]
            while curr_sum > residual and left <= right:
                curr_sum -= nums_doubled[left]
                left +=1
            if curr_sum == residual:
                min_length = min(min_length, right - left +1)
                # Move left to find shorter subarrays
                curr_sum -= nums_doubled[left]
                left +=1
        if min_length == float('inf'):
            return -1
        return k * N + min_length