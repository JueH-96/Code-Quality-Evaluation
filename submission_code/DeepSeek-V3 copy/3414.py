class Solution:
    def waysToReachStair(self, k: int) -> int:
        if k == 0:
            return 2
        if k == 1:
            return 4
        
        # Initialize the number of ways
        ways = 0
        
        # Iterate over possible jump counts
        jump = 0
        while True:
            # Calculate the maximum possible position after jumps
            max_pos = 1
            for j in range(jump):
                max_pos += 2 ** j
            if max_pos < k:
                jump += 1
                continue
            # Now, find the number of ways to reach k with this jump count
            # The position after jumps is 1 + sum_{i=0}^{jump-1} 2^i = 2^jump
            pos = 2 ** jump
            if pos == k:
                ways += 1
            elif pos > k:
                # To reach k, we need to go down (pos - k) steps
                # But we cannot go down consecutively
                # The number of ways is the number of ways to arrange the down steps
                # such that no two are consecutive
                # This is equivalent to the number of ways to partition (pos - k) into
                # parts of size 1, with at least one step in between
                # This is the same as the number of ways to arrange (pos - k) down steps
                # with at least one up step in between
                # The number of ways is 2^(number of down steps - 1)
                down_steps = pos - k
                if down_steps == 1:
                    ways += 1
                else:
                    ways += 2 ** (down_steps - 1)
            jump += 1
            if 2 ** jump > 10**18:
                break
        return ways