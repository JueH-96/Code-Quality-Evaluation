class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        
        for num in nums:
            # Check if num can be expressed as x OR (x + 1)
            # This happens when num has the form of consecutive 1s from right
            # or has a specific pattern
            
            # Find the rightmost 0 bit in num
            # If num is all 1s from the right, then we can find x
            
            # A number can be expressed as x OR (x+1) if and only if
            # it doesn't have the pattern 10 (a 1 followed immediately by 0)
            # when reading from right to left
            
            found = False
            for x in range(num):
                if (x | (x + 1)) == num:
                    ans.append(x)
                    found = True
                    break
            
            if not found:
                ans.append(-1)
        
        return ans