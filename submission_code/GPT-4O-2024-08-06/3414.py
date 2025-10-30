class Solution:
    def waysToReachStair(self, k: int) -> int:
        # The number of ways to reach stair k is 2^(k+1)
        # This is because for each stair from 0 to k, Alice can either:
        # - Go up using the jump operation, which doubles the number of ways
        # - Go down using the down operation, which also doubles the number of ways
        # Therefore, for each stair, there are two choices, leading to 2^(k+1) ways in total.
        
        return 2 ** (k + 1)