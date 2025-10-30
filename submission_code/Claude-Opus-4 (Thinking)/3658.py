class Solution:
    def minDifference(self, nums: List[int]) -> int:
        from collections import defaultdict, deque
        
        n = len(nums)
        
        # If no -1s, return the max difference
        if -1 not in nums:
            return max(abs(nums[i] - nums[i-1]) for i in range(1, n))
        
        # If all are -1s, we can make them all equal
        if all(num == -1 for num in nums):
            return 0
        
        # Binary search on the maximum difference
        def canAchieve(target):
            # Check if we can achieve max difference <= target
            
            # First, check existing differences between non-(-1) elements
            for i in range(1, n):
                if nums[i-1] != -1 and nums[i] != -1:
                    if abs(nums[i] - nums[i-1]) > target:
                        return False
            
            # Find valid ranges for each -1 position
            ranges = []
            for i in range(n):
                if nums[i] == -1:
                    lo, hi = 1, 10**9
                    
                    # Constraint from left neighbor
                    if i > 0 and nums[i-1] != -1:
                        lo = max(lo, nums[i-1] - target)
                        hi = min(hi, nums[i-1] + target)
                    
                    # Constraint from right neighbor
                    if i < n-1 and nums[i+1] != -1:
                        lo = max(lo, nums[i+1] - target)
                        hi = min(hi, nums[i+1] + target)
                    
                    if lo > hi:
                        return False
                    
                    ranges.append((lo, hi, i))
            
            # Build conflict graph for adjacent -1s
            adj = defaultdict(list)
            for i in range(len(ranges)):
                for j in range(i + 1, len(ranges)):
                    if ranges[j][2] == ranges[i][2] + 1:  # Adjacent positions
                        # Check if their ranges intersect
                        if max(ranges[i][0], ranges[j][0]) > min(ranges[i][1], ranges[j][1]):
                            # No intersection, they must have different values
                            adj[i].append(j)
                            adj[j].append(i)
            
            # Try to 2-color the graph
            color = [-1] * len(ranges)
            
            for start in range(len(ranges)):
                if color[start] == -1:
                    # BFS to color the component
                    queue = deque([start])
                    color[start] = 0
                    
                    while queue:
                        u = queue.popleft()
                        for v in adj[u]:
                            if color[v] == -1:
                                color[v] = 1 - color[u]
                                queue.append(v)
                            elif color[v] == color[u]:
                                # Same color but they're connected (conflict)
                                return False
            
            # Verify that each color group has a valid common value
            for c in [0, 1]:
                group_ranges = [ranges[i][:2] for i in range(len(ranges)) if color[i] == c]
                if group_ranges:
                    common_lo = max(r[0] for r in group_ranges)
                    common_hi = min(r[1] for r in group_ranges)
                    if common_lo > common_hi:
                        return False
            
            return True
        
        # Binary search for the answer
        left, right = 0, 10**9
        while left < right:
            mid = (left + right) // 2
            if canAchieve(mid):
                right = mid
            else:
                left = mid + 1
        
        return left