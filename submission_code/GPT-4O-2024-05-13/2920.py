class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        n = len(nums)
        if n == 1:
            return 0
        
        # Dictionary to store the last seen index of each number
        last_seen = defaultdict(lambda: -1)
        # Dictionary to store the maximum distance between consecutive occurrences of each number
        max_distance = defaultdict(int)
        
        for i in range(n):
            num = nums[i]
            if last_seen[num] != -1:
                distance = (i - last_seen[num]) // 2
                max_distance[num] = max(max_distance[num], distance)
            last_seen[num] = i
        
        # To handle the circular nature of the array
        for num in last_seen:
            distance = (n + last_seen[num] - last_seen[num]) // 2
            max_distance[num] = max(max_distance[num], distance)
        
        # The minimum number of seconds needed is the minimum of the maximum distances
        return min(max_distance.values())