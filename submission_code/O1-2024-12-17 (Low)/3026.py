class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        
        # We can pick at most floor(target/2) distinct small positive numbers
        # from 1, 2, ... (since picking x forbids picking target - x, if both < target).
        # Once we pick those, if we still need more numbers, we pick starting from 'target'
        # upwards, as they cannot conflict with smaller numbers for the given sum condition.
        
        # Let k = min(n, target // 2). We first pick the integers 1..k.
        # Then we pick the next (n - k) integers: target, target+1, ..., target+(n - k) - 1.
        
        # sum(1..k) = k*(k+1)//2
        # sum(target..target+m-1) = m*target + sum(0..m-1) = m*target + m*(m-1)//2, where m = (n - k).
        
        k = min(n, target // 2)
        m = n - k
        
        # Function to safely compute x*(x+1)//2 mod
        def sum_first_x(x):
            return (x * (x + 1) // 2) % MOD
        
        # Sum of the first k natural numbers
        sum_k = sum_first_x(k)
        
        # Sum of m consecutive numbers starting from 'target'
        # = m*target + sum(0..m-1) = m*target + m*(m-1)//2
        # We'll do each part modulo and combine.
        part1 = (m * target) % MOD
        part2 = sum_first_x(m - 1) if m > 0 else 0  # sum(0..m-1) = sum(1..m-1)
        sum_m_from_target = (part1 + part2) % MOD
        
        return (sum_k + sum_m_from_target) % MOD