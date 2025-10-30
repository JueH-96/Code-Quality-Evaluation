from typing import List, Tuple
import sys
sys.setrecursionlimit(10000)

class Solution:
    def minCost(self, nums: List[int]) -> int:
        n = len(nums)
        # Use memoization for DP states.
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dp(pos: int, buf: Tuple[int, ...]) -> int:
            buf = list(buf)
            # refill buffer (simulate that buf holds the current leading elements from the original sequence).
            while len(buf) < 3 and pos < n:
                buf.append(nums[pos])
                pos += 1
            
            k = len(buf)
            # if fewer than 3 elements remain and we have consumed nums, then we remove them all in one op.
            if pos == n and k < 3:
                if k == 0:
                    return 0
                else:
                    return max(buf)
            
            # At this point, buf always has exactly 3 elements.
            # There are 3 choices: remove any pair from the first three elements.
            best = float('inf')
            # option 1: remove buf[0] and buf[1]
            cost1 = max(buf[0], buf[1])
            new_buf = (buf[2],)
            best = min(best, cost1 + dp(pos, new_buf))
            # option 2: remove buf[0] and buf[2]
            cost2 = max(buf[0], buf[2])
            new_buf = (buf[1],)
            best = min(best, cost2 + dp(pos, new_buf))
            # option 3: remove buf[1] and buf[2]
            cost3 = max(buf[1], buf[2])
            new_buf = (buf[0],)
            best = min(best, cost3 + dp(pos, new_buf))
            return best
            
        return dp(0, ())
    
# Below are some tests:
if __name__ == '__main__':
    sol = Solution()
    # Example 1
    print(sol.minCost([6,2,8,4]))  # Expected output 12
    # Example 2
    print(sol.minCost([2,1,3,3]))  # Expected output 5
    # Additional tests:
    print(sol.minCost([5]))       # Only one element, cost should be 5.
    print(sol.minCost([1,2,3]))   # Three elements, removal cost = max(1,2,3)? Let's simulate options:
                                  # Actually, when there are exactly 3 elements (buf filled fully but pos==n),
                                  # our dp always treats state len(buf)==3 as choices:
                                  # Option1: remove 1,2 cost 2 and then remove leftover [3] cost 3 ==> 5.
                                  # Option2: remove 1,3 cost 3 and then remove leftover [2] cost 2 ==> 5.
                                  # Option3: remove 2,3 cost 3 and then remove leftover [1] cost 1 ==> 4.
                                  # So answer should be 4.
    print(sol.minCost([1,2,3]))