class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        gcd_result = nums[0]
        for num in nums[1:]:
            gcd_result = math.gcd(gcd_result, num)
        
        min_length = 0
        for num in nums:
            if math.gcd(num, gcd_result) == gcd_result:
                min_length += 1
            if gcd_result == 1:
                break
        
        return 1 if min_length > 0 else float('inf')