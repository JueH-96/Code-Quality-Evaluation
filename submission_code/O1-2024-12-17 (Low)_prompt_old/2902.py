class Solution:
    def maxSum(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        def get_max_digit(n: int) -> int:
            return max(int(d) for d in str(n))
        
        digit_map = defaultdict(list)
        for num in nums:
            d = get_max_digit(num)
            digit_map[d].append(num)
        
        max_sum = -1
        for d, arr in digit_map.items():
            if len(arr) > 1:
                arr.sort(reverse=True)
                max_sum = max(max_sum, arr[0] + arr[1])
        
        return max_sum