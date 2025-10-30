class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        infinity = 10 ** 9
        n = len(nums)
        total = sum(nums)
        if target%total == 0:
            ans = target//total * n + 0
        else:
            ans = target//total * n + infinity
        
        s = 0
        e = 0
        total_cur = 0
        while e<2*n:
            total_cur += nums[e % n]
            while total_cur > target%total:
                total_cur -= nums[s % n]
                s += 1
            if total_cur == target%total:
                ans = min(ans,e - s + 1)
            e += 1
                
        return ans if ans!= infinity else -1