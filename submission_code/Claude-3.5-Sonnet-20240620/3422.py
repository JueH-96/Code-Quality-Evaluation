class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Initialize the array
        a = [1] * n
        
        # Simulate k seconds
        for _ in range(k):
            # Create a new array for the next state
            new_a = [0] * n
            
            # Update each element
            for i in range(n):
                if i == 0:
                    new_a[i] = a[i]
                else:
                    new_a[i] = (new_a[i-1] + a[i]) % MOD
            
            # Update the original array
            a = new_a
        
        # Return the last element
        return a[n-1]