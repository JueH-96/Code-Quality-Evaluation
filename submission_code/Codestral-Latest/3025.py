class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        from collections import deque

        # Helper function to check if a subsequence sum can be achieved
        def can_achieve_sum(nums, target):
            dp = [False] * (target + 1)
            dp[0] = True
            for num in nums:
                for j in range(target, num - 1, -1):
                    dp[j] = dp[j] or dp[j - num]
            return dp[target]

        # If the target can be achieved without any operations, return 0
        if can_achieve_sum(nums, target):
            return 0

        # Initialize a queue for BFS
        queue = deque([(nums, 0)])
        visited = set(tuple(nums))

        while queue:
            current_nums, steps = queue.popleft()

            # Try to split each element greater than 1
            for i in range(len(current_nums)):
                if current_nums[i] > 1:
                    new_nums = current_nums[:i] + [current_nums[i] // 2, current_nums[i] // 2] + current_nums[i+1:]
                    new_tuple = tuple(new_nums)

                    if new_tuple not in visited:
                        visited.add(new_tuple)
                        if can_achieve_sum(new_nums, target):
                            return steps + 1
                        queue.append((new_nums, steps + 1))

        return -1