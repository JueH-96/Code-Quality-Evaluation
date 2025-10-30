class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        queue = [(0, 0)]  # (operations, index)
        seen = {0: 0}
        
        while queue:
            operations, index = heapq.heappop(queue)
            if index == len(nums):
                return operations
            
            key = index * 10**9 + (nums[index] - target[index])
            if key not in seen or seen[key] > operations:
                seen[key] = operations
                heapq.heappush(queue, (operations + abs(nums[index] - target[index]), index + 1))
                
            key = index * 10**9 + (nums[index - 1] - target[index] if index else nums[index] - target[index])
            if key not in seen or seen[key] > operations + 1:
                new_nums = nums[:]
                if index: new_nums[index - 1] += nums[index] - target[index]
                seen[key] = operations + 1
                heapq.heappush(queue, (operations + 1, index + 1))
                
            if index == len(nums) - 1:
                break
                
        return seen[index * 10**9 + target[index] - (index < len(nums) - 1 and nums[index + 1] - target[index] or 0)]