class Solution:
    def minOperations(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count the frequency of each distinct number
        freq = Counter(nums)
        
        def cost_for_count(n: int) -> int:
            """
            Returns the minimum number of operations to remove n identical elements
            using groups of 2 or 3 (pairs or triples), or -1 if impossible.
            
            We want to solve 3*x + 2*y = n minimizing (x + y).
            This is equivalent to picking the largest x (number of triples)
            such that n - 3*x is nonnegative and divisible by 2.
            """
            # The maximum number of triples we can take is n // 3
            x_max = n // 3
            
            # We only need to check at most 3 consecutive values for x 
            # (from x_max down to x_max - 2) because each step changes the remainder mod 2.
            for x in range(x_max, x_max - 3, -1):
                if x < 0:
                    break
                remaining = n - 3 * x
                if remaining >= 0 and remaining % 2 == 0:
                    # y = remaining // 2
                    # The cost is x + y
                    return x + (remaining // 2)
            
            return -1
        
        total_operations = 0
        # Calculate the cost for each distinct value
        for val, count in freq.items():
            c = cost_for_count(count)
            if c == -1:
                return -1
            total_operations += c
        
        return total_operations