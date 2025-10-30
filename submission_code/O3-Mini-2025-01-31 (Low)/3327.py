from typing import List

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        res = float('inf')
        # For each candidate starting index (aliceIndex)
        for i in range(n):
            # Determine how many ones are picked for free initially.
            picked = 1 if nums[i] == 1 else 0
            needed = k - picked  # remaining ones required
            # bounds: how many steps available to left and right
            leftAvailable = i      # positions: 0 ... i-1
            rightAvailable = n - i - 1  # positions: i+1 ... n-1
            # Precompute left arrays: leftCost[r] and leftConv[r] for r from 0 to leftAvailable
            leftCost = [0]
            leftConv = [0]
            for r in range(1, leftAvailable + 1):
                idx = i - r
                # If cell is in bounds, cost is:
                if nums[idx] == 1:
                    leftCost.append(leftCost[-1] + 1)
                    leftConv.append(leftConv[-1])
                else:
                    leftCost.append(leftCost[-1] + 2)
                    leftConv.append(leftConv[-1] + 1)
            # Precompute right arrays similarly for r from 0 to rightAvailable
            rightCost = [0]
            rightConv = [0]
            for r in range(1, rightAvailable + 1):
                idx = i + r
                if nums[idx] == 1:
                    rightCost.append(rightCost[-1] + 1)
                    rightConv.append(rightConv[-1])
                else:
                    rightCost.append(rightCost[-1] + 2)
                    rightConv.append(rightConv[-1] + 1)
                    
            # For the required ones (needed picks), try every possible division:
            # Let L be number of picks taken from the left and R = needed - L from the right.
            local_best = float('inf')
            # It is possible that needed is 0 (if nums[i]==1 and k==1)
            if needed == 0:
                local_best = 0
            else:
                # L can range from 0 to needed. But also cannot exceed leftAvailable.
                for L in range(0, needed+1):
                    R = needed - L
                    if L <= leftAvailable and R <= rightAvailable:
                        conv_used = leftConv[L] + rightConv[R]
                        if conv_used <= maxChanges:
                            moves = leftCost[L] + rightCost[R]
                            if moves < local_best:
                                local_best = moves
            if local_best < res:
                res = local_best
        # In worst-case, answer is res plus 0 (initial pickup is free).
        return res if res != float('inf') else -1

# For testing
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    nums = [1,1,0,0,0,1,1,0,0,1]
    k = 3
    maxChanges = 1
    print(sol.minimumMoves(nums, k, maxChanges))  # Expected output: 3

    # Example 2:
    nums = [0,0,0,0]
    k = 2
    maxChanges = 3
    print(sol.minimumMoves(nums, k, maxChanges))  # Expected output: 4