class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Since the only allowed operation is to decrease some values,
        # if any number is less than k, then it is impossible to reach k.
        for num in nums:
            if num < k:
                return -1
        
        # Every operation can only target the current maximum group.
        # The optimal strategy is to lower all numbers equal to the current maximum down to
        # the next distinct value until all numbers are k.
        # This means that for every distinct integer in nums that is strictly greater than k,
        # we need one operation.
        distinct_above = {num for num in nums if num > k}
        return len(distinct_above)