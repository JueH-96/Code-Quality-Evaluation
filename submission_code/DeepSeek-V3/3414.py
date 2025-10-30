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
            # Calculate the target stair after the jump
            target = 1 + (1 << jump)
            if target > k:
                break
            # Calculate the number of steps needed to reach k from target
            steps = k - target
            if steps < 0:
                jump += 1
                continue
            # The number of ways is 2^steps
            ways += (1 << steps)
            jump += 1
        
        # Add the case where no jumps are used
        if k == 1:
            ways += 1
        else:
            ways += 1 if k == 0 else 0
        
        return ways