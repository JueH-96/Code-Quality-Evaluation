class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        diff = [0] * n
        for i in range(n):
            diff[i] = nums[i]
        
        for i in range(n):
            if diff[i] > 0:
                if i + k > n:
                    return False
                for j in range(i, i + k):
                    diff[j] -= diff[i]
            elif diff[i] < 0:
                return False
        
        return all(x == 0 for x in diff)