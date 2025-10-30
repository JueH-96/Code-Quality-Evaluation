class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        # Define the desired target as all '0's or all '1's
        # We will compute the cost for both and take the minimum
        
        def compute_cost(target):
            dp_no_flip = 0
            dp_flip = float('inf')
            
            for i in range(n):
                bit = s[i]
                # If no flip, check if current bit matches target
                if bit == target:
                    new_dp_no_flip = dp_no_flip
                else:
                    new_dp_no_flip = float('inf')
                
                # If flip, check if flipped bit matches target
                flipped_bit = '1' if bit == '0' else '0'
                if flipped_bit == target:
                    new_dp_flip = dp_no_flip + min(i + 1, n - i)
                else:
                    new_dp_flip = float('inf')
                
                # Update dp_no_flip and dp_flip
                dp_no_flip, dp_flip = new_dp_no_flip, new_dp_flip
            
            return dp_no_flip if dp_no_flip != float('inf') else dp_flip
        
        # Compute cost for making all '0's and all '1's
        cost_0 = compute_cost('0')
        cost_1 = compute_cost('1')
        
        return min(cost_0, cost_1)