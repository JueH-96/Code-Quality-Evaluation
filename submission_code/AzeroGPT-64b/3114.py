class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        def getMax(nums):
            nums = [0] + nums + [0]
            st = []
            val = 0
            for i, x in enumerate(nums):
                while st and nums[st[-1]] >= x:
                    val += (nums[st.pop()]-x) * (i - st[-1] - 1)
                val += nums[i] * (1 + len(st))
                st.append(i)
            return val
        
        return max(getMax(maxHeights), getMax(maxHeights[::-1]))