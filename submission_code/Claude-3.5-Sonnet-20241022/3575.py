class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_val = 0
        
        def get_value(seq):
            x = len(seq) // 2
            first_half = 0
            second_half = 0
            
            # OR operation for first half
            for i in range(x):
                first_half |= seq[i]
                
            # OR operation for second half
            for i in range(x, 2*x):
                second_half |= seq[i]
                
            # XOR of the two halves
            return first_half ^ second_half
        
        def backtrack(start, curr_seq):
            nonlocal max_val
            
            if len(curr_seq) == 2 * k:
                max_val = max(max_val, get_value(curr_seq))
                return
            
            for i in range(start, n):
                curr_seq.append(nums[i])
                backtrack(i + 1, curr_seq)
                curr_seq.pop()
        
        backtrack(0, [])
        return max_val