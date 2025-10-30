class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        def check(num, target):
            # Check if num OR (num + 1) equals target
            return (num | (num + 1)) == target
        
        def find_min_value(target):
            # Try values from 0 to target to find the minimum value that satisfies the condition
            for i in range(target):
                if check(i, target):
                    return i
            return -1
        
        ans = []
        for num in nums:
            min_val = find_min_value(num)
            ans.append(min_val)
            
        return ans