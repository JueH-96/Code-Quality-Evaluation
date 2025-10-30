class Solution:
    def minArraySum(self, nums: list[int], k: int, op1: int, op2: int) -> int:
        from collections import defaultdict
        
        dp = {(op1, op2): 0}
        for x in nums:
            temp_dp = defaultdict(lambda: float('inf'))
            for (a, b), current_sum in dp.items():
                # Option 0: Do nothing
                new_sum = current_sum + x
                if new_sum < temp_dp[(a, b)]:
                    temp_dp[(a, b)] = new_sum
                
                # Option 1: Apply Operation 1
                if a > 0:
                    new_val = (x + 1) // 2
                    new_sum = current_sum + new_val
                    new_a = a - 1
                    new_b = b
                    if new_sum < temp_dp[(new_a, new_b)]:
                        temp_dp[(new_a, new_b)] = new_sum
                
                # Option 2: Apply Operation 2
                if b > 0 and x >= k:
                    new_val = x - k
                    new_sum = current_sum + new_val
                    new_a = a
                    new_b = b - 1
                    if new_sum < temp_dp[(new_a, new_b)]:
                        temp_dp[(new_a, new_b)] = new_sum
                
                # Option 3: Apply Operation 1 then Operation 2
                if a > 0 and b > 0:
                    after_op1 = (x + 1) // 2
                    if after_op1 >= k:
                        new_val = after_op1 - k
                        new_sum = current_sum + new_val
                        new_a = a - 1
                        new_b = b - 1
                        if new_sum < temp_dp[(new_a, new_b)]:
                            temp_dp[(new_a, new_b)] = new_sum
                
                # Option 4: Apply Operation 2 then Operation 1
                if a > 0 and b > 0 and x >= k:
                    after_op2 = x - k
                    new_val = (after_op2 + 1) // 2
                    new_sum = current_sum + new_val
                    new_a = a - 1
                    new_b = b - 1
                    if new_sum < temp_dp[(new_a, new_b)]:
                        temp_dp[(new_a, new_b)] = new_sum
            
            # Convert defaultdict to a regular dict for the next iteration
            dp = {key: val for key, val in temp_dp.items()}
        
        return min(dp.values()) if dp else 0