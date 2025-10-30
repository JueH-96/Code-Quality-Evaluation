class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        # Create a list of all possible values for each number after operations
        possibilities = []
        
        for num in nums:
            current = set()
            # Original number
            current.add(num)
            
            # After operation 1 (divide by 2 and round up)
            div_val = (num + 1) // 2
            current.add(div_val)
            
            # After operation 2 (subtract k)
            if num >= k:
                current.add(num - k)
                # After both operations
                if div_val >= k:
                    current.add(div_val - k)
            
            possibilities.append(sorted(list(current)))
        
        # dp[i][j][l] represents minimum sum possible using first i numbers,
        # j operation1s and l operation2s
        dp = {}
        
        def solve(index, op1_left, op2_left):
            if index == n:
                return 0
                
            state = (index, op1_left, op2_left)
            if state in dp:
                return dp[state]
            
            curr_min = float('inf')
            num = nums[index]
            
            # Try all possibilities for current number
            # 1. No operation
            curr_min = min(curr_min, num + solve(index + 1, op1_left, op2_left))
            
            # 2. Operation 1 (if available)
            if op1_left > 0:
                div_val = (num + 1) // 2
                curr_min = min(curr_min, div_val + solve(index + 1, op1_left - 1, op2_left))
            
            # 3. Operation 2 (if available and possible)
            if op2_left > 0 and num >= k:
                curr_min = min(curr_min, num - k + solve(index + 1, op1_left, op2_left - 1))
            
            # 4. Both operations (if both available and operation 2 is possible after operation 1)
            if op1_left > 0 and op2_left > 0:
                div_val = (num + 1) // 2
                if div_val >= k:
                    curr_min = min(curr_min, div_val - k + solve(index + 1, op1_left - 1, op2_left - 1))
            
            dp[state] = curr_min
            return curr_min
        
        return solve(0, op1, op2)