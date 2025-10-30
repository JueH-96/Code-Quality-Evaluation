from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        max_num = 10**6
        spf = list(range(max_num + 1))  # Initialize smallest prime factors
        
        # Sieve of Eratosthenes to compute smallest prime factors
        for i in range(2, int(max_num**0.5) + 1):
            if spf[i] == i:  # i is a prime
                for j in range(i * i, max_num + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        n = len(nums)
        if n == 0:
            return 0
        
        processed = [0] * n
        processed[-1] = nums[-1]
        total_steps = 0
        
        for i in range(n - 2, -1, -1):
            current_val = nums[i]
            next_val = processed[i + 1]
            current = current_val
            steps = 0
            possible = True
            prev = None
            
            while current > next_val:
                if current == 1:
                    possible = False
                    break
                # Check if current is a prime
                if spf[current] == current:
                    possible = False
                    break
                # Compute greatest proper divisor
                g = current // spf[current]
                current = current // g
                steps += 1
                # Check for infinite loop (current not changing)
                if current == prev:
                    if current > next_val:
                        possible = False
                    break
                prev = current
            
            if not possible or current > next_val:
                return -1
            processed[i] = current
            total_steps += steps
        
        return total_steps