class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        a = [1] * n  # Initialize array with all 1s
        
        for _ in range(k):
            # Create the next state
            new_a = [1] * n  # a[0] always remains 1
            running_sum = 1  # Keep track of prefix sum
            
            for i in range(1, n):
                # New value is sum of all preceding elements plus itself
                new_a[i] = (running_sum + a[i]) % MOD
                # Update running sum for next iteration
                running_sum = (running_sum + a[i]) % MOD
            
            a = new_a  # Update array for next second
        
        return a[n-1]  # Return value of last element