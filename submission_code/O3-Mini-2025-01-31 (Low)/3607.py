from typing import List
import math

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        # Pre-calculate the chain for each number.
        # For each number x, we get a list of (value, cost) pairs that can be achieved
        # by applying the given operation repeatedly.
        def get_chain(x: int):
            # For number 1, no operation can be performed.
            chain = [(x, 0)]
            # If x is 1, chain is just [(1,0)].
            if x == 1:
                return chain
            steps = 0
            current = x
            while True:
                steps += 1
                # Compute the greatest proper divisor for current
                # For current > 1, the greatest proper divisor is:
                # If current is composite, it is current // smallest_factor,
                # and if current is prime then its only proper divisor is 1.
                spf = None
                # Find the smallest factor greater than 1:
                # We only need to check up to sqrt(current)
                # (if none found, then current is prime)
                r = int(math.isqrt(current))
                for i in range(2, r+1):
                    if current % i == 0:
                        spf = i
                        break
                if spf is None:  # current is prime, its only proper divisor is 1
                    next_val = current // 1
                else:
                    next_val = current // spf
                
                # If the operation does not change the number, then no further progress.
                if next_val == current:
                    break
                # Avoid producing a chain value that is higher than before
                # and add the new candidate.
                chain.append((next_val, steps))
                # If we've reached 1 then it's the bottom of the chain.
                if next_val == 1:
                    break
                current = next_val
            return chain

        n = len(nums)
        # For each number in nums compute its chain
        chains = [get_chain(num) for num in nums]
        
        # dp will track the minimal cost to achieve a sequence ending with some last value.
        # We'll use a dictionary mapping last_value -> minimal total operations cost.
        # For the 0-th element, we populate dp with every candidate from its chain.
        dp = {}
        for val, cost in chains[0]:
            # There can be duplicate values but we keep the minimal cost.
            if val in dp:
                dp[val] = min(dp[val], cost)
            else:
                dp[val] = cost
        
        # Process the rest of the numbers.
        for i in range(1, n):
            new_dp = {}
            # For each candidate option (candidate value, cost) in the chain of nums[i]
            for curr_val, curr_cost in chains[i]:
                # We can attach this candidate only if there is a previous dp state with value <= curr_val.
                # We'll iterate over dp keys.
                for prev_val, prev_cost in dp.items():
                    if prev_val <= curr_val:
                        total_cost = prev_cost + curr_cost
                        if curr_val in new_dp:
                            new_dp[curr_val] = min(new_dp[curr_val], total_cost)
                        else:
                            new_dp[curr_val] = total_cost
            dp = new_dp
            if not dp:
                return -1  # no way to have a non-decreasing sequence
        
        # The answer is the minimum total cost among all possible final candidate values.
        return min(dp.values())

# Simple testing code:
if __name__ == '__main__':
    sol = Solution()
    # Example 1: [25, 7] -> 1
    print(sol.minOperations([25, 7]))   # Expected output: 1
    # Example 2: [7, 7, 6] -> -1
    print(sol.minOperations([7, 7, 6]))   # Expected output: -1
    # Example 3: [1, 1, 1, 1] -> 0
    print(sol.minOperations([1, 1, 1, 1]))  # Expected output: 0