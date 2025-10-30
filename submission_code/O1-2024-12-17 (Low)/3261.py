class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        """
        We want to apply up to k "merge" operations, each operation replacing two adjacent
        elements (say nums[i], nums[i+1]) by a single element (nums[i] & nums[i+1]). 
        This reduces the array length by 1 per merge. After at most k merges, the array 
        has length at least n - k. We want the bitwise-OR of the resulting elements 
        to be as small as possible.

        KEY INSIGHT:
          - If we end up with m = n - k elements, those m elements are each the AND of some
            contiguous sub-subarray of the original array (since every merge is between
            adjacent elements, effectively "gluing" them).
          - Let those final m = n-k elements be A1, A2, ..., A_m, each being an AND of a
            consecutive block in nums. The result we want is (A1 OR A2 OR ... OR A_m).

        OBSERVATION:
          - "Minimize (A1 OR A2 OR ... OR A_m)" is equivalent to ensuring each Ai is
            as free as possible of high bits, because (bitwise OR) accumulates bits from
            anywhere.
          - Each Ai = AND of some block of the array. The AND of a block is ≤ the minimum
            element in that block, and typically "shrinks" bitwise as we add more elements.

        APPROACH (Binary Search on the Final OR):
          1) We binary-search on the integer X (0 to (2^30 - 1)) to check "Can the final OR
             be ≤ X?" If yes, we try lower; if no, we go higher.
          2) To check feasibility "Can we make the final OR ≤ X with ≤ k merges?" is 
             equivalent to: "Can we form at most (n - k) contiguous groups so that
             the AND of each group is ≤ X?" because if all group-ANDs are ≤ X, their OR
             will be ≤ X as well.

          Checking feasibility "can(X)":
            - We'll greedily partition nums into subarrays (from left to right).
            - Maintain a running AND for the current group. We iterate over nums:
                current_and &= nums[i]
                if current_and > X, we must start a new group at i (because the current
                group can't stay ≤ X if we include nums[i]).
                  - That means end the previous group at i-1, and current_and resets 
                    to nums[i].
                  - If nums[i] alone is > X, it's impossible immediately (fail).
            - Count how many groups are formed this way. If the group-count ≤ (n - k),
              it means we used at most k merges (forming at most n-k blocks). So it's feasible.

        COMPLEXITY:
          - Each feasibility check is O(n).
          - We do up to ~30 steps of binary search (since nums[i] < 2^30).
          - Overall O(n log(2^30)) which is acceptable for n up to 1e5.

        Let's implement it.
        """

        import math

        n = len(nums)

        # Quick edge case: if k == 0, we cannot merge at all, so just return OR of nums
        # (though the binary-search method will also handle it fine, but let's short-circuit).
        if k == 0:
            ans = 0
            for v in nums:
                ans |= v
            return ans

        # Function to check feasibility: "Can we get final OR <= x with <= k merges?"
        # Equivalently: "Can we partition into at most (n - k) groups where each group's AND <= x?"
        def can(x: int) -> bool:
            group_count = 1
            current_and = nums[0]
            if current_and > x:
                return False  # single element already too big

            for i in range(1, n):
                new_and = current_and & nums[i]
                if new_and > x:
                    # Need to start a new group at i
                    group_count += 1
                    current_and = nums[i]
                    if current_and > x:
                        return False
                else:
                    current_and = new_and
            return group_count <= (n - k)

        # We'll do a binary-search on [0, (1 << 30) - 1]
        left, right = 0, (1 << 30) - 1
        answer = right

        while left <= right:
            mid = (left + right) // 2
            if can(mid):
                answer = mid
                right = mid - 1
            else:
                left = mid + 1

        return answer