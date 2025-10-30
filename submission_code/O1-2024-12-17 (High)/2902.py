class Solution:
    def maxSum(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        # Function to find the maximum digit of a number
        def max_digit(x):
            return max(int(d) for d in str(x))
        
        # Group numbers by their maximum digit
        groups = defaultdict(list)
        for num in nums:
            groups[max_digit(num)].append(num)
        
        # For each group, find the two largest numbers and record their sum
        answer = -1
        for digit in groups:
            if len(groups[digit]) > 1:
                groups[digit].sort(reverse=True)
                curr_sum = groups[digit][0] + groups[digit][1]
                answer = max(answer, curr_sum)
        
        return answer