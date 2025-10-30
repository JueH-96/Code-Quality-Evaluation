class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        N = len(nums)
        nums.append(float('inf'))  # Sentinel for easier computation of Right[]
        Left = [-1]*N
        Right = [N]*N

        stack = []
        # Compute Left[i]
        for i in range(N):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            Left[i] = stack[-1] if stack else -1
            stack.append(i)
        
        stack = []
        # Compute Right[i]
        for i in range(N-1, -1, -1):
            while stack and nums[stack[-1]] < nums[i]:
                stack.pop()
            Right[i] = stack[-1] if stack else N
            stack.append(i)
        
        from collections import defaultdict

        # Group positions by intervals where nums[i] is maximum
        intervals = defaultdict(list)
        for i in range(N):
            l = Left[i]+1
            r = Right[i]-1
            intervals[(l, r)].append(i)
        
        ans = 0
        # For each interval, count the number of valid subarrays
        for (l, r), indices in intervals.items():
            value_positions = defaultdict(list)
            for idx in indices:
                v = nums[idx]
                value_positions[v].append(idx)
            for positions in value_positions.values():
                m = len(positions)
                ans += m * (m+1) // 2  # Number of subarrays
        return ans