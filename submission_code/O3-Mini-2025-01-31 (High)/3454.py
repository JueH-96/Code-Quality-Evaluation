class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        # The key idea is to work with the differences between target and nums.
        # Let d[i] = target[i] - nums[i]. We need to “apply” operations so that 
        # each index i in nums gets increased (or decreased) by d[i].
        #
        # Each allowed operation chooses a contiguous subarray and adds either +1 or -1.
        # In terms of range updates, an operation (L, R, s) adds s to every element
        # between L and R. This operation can be “encoded” in a difference array g of length n+1:
        #   g[L]   += s
        #   g[R+1] -= s
        #
        # Now, if we set up an array d such that:
        #   d[0] = target[0] - nums[0]
        #   d[i] = target[i] - nums[i] for i >= 0,
        # then we can recover d from its “gradient” by:
        #   d[0] = g[0]
        #   d[i] = d[i-1] + g[i]   for 1 <= i < n.
        #
        # In other words, we can derive:
        #   g[0] = d[0]
        #   For i = 1 to n-1, g[i] = d[i] - d[i-1]
        #   And to “balance” the update for the whole array we set g[n] = -d[n-1]
        #
        # Every allowed operation contributes +1 at some index L and -1 at index R+1,
        # so each operation contributes a total "cost" of 2 (in absolute value) to the
        # sum of absolute values of the vector g.
        #
        # Thus, the minimum number of operations required is exactly:
        #
        #    (|g[0]| + sum(|g[1]|,...,|g[n-1]|) + |g[n]|) / 2
        #
        # where:
        #   g[0] = d[0] = target[0] - nums[0]
        #   g[i] = (target[i] - nums[i]) - (target[i-1] - nums[i-1]) for 1 <= i < n
        #   g[n] = - (target[n-1] - nums[n-1])
        #
        # Let’s check quickly on the examples:
        # Example 1:
        #   nums = [3,5,1,2], target = [4,6,2,4]
        #   d = [1, 1, 1, 2]
        #   g[0] = 1, g[1] = 1-1 = 0, g[2] = 1-1 = 0, g[3] = 2-1 = 1, g[4] = -2.
        #   Sum of absolute values = |1| + 0 + 0 + |1| + |2| = 1 + 1 + 2 = 4, so answer = 4//2 = 2.
        #
        # Example 2:
        #   nums = [1,3,2], target = [2,1,4]
        #   d = [1, -2, 2]
        #   g[0] = 1, g[1] = (-2)-1 = -3, g[2] = 2 - (-2) = 4, g[3] = -2.
        #   Sum of absolute values = 1 + 3 + 4 + 2 = 10, so answer = 10//2 = 5.
        #
        # Now we implement this strategy.
        
        n = len(nums)
        # d[0] = target[0] - nums[0]
        d0 = target[0] - nums[0]
        total_change = abs(d0)
        prev = d0
        
        # For i from 1 to n-1, compute g[i] = (target[i]-nums[i]) - (target[i-1]-nums[i-1])
        for i in range(1, n):
            d = target[i] - nums[i]
            diff = d - prev  # This is g[i]
            total_change += abs(diff)
            prev = d
        
        # Finally, account for the last boundary: g[n] = - d[n-1]
        total_change += abs(prev)
        
        # Each operation adds a total absolute "change" of 2 in g,
        # so the minimal number of operations is half of total_change.
        return total_change // 2

# Example testing:
if __name__ == '__main__':
    sol = Solution()
    print(sol.minimumOperations([3,5,1,2], [4,6,2,4]))  # Expected output: 2
    print(sol.minimumOperations([1,3,2], [2,1,4]))        # Expected output: 5