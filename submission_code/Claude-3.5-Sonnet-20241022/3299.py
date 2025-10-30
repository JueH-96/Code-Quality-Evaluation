class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        count = Counter(nums)
        result = 1
        
        # Handle 1 separately
        if 1 in count:
            if count[1] % 2 == 0:
                result = max(result, count[1] - 1)
            else:
                result = max(result, count[1])
            
        # For each number except 1
        for num in count:
            if num == 1:
                continue
                
            curr_len = 0
            curr = num
            
            # Check sequence of squares
            while curr in count and count[curr] >= 2:
                curr_len += 2
                curr = curr * curr
                
            # Add last number if it exists once
            if curr in count and count[curr] >= 1:
                curr_len += 1
                
            result = max(result, curr_len)
            
        return result