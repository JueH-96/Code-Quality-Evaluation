class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        answer = []
        
        for q in queries:
            target = nums[q]
            min_distance = float('inf')
            
            # Search for the target value in the circular array
            for i in range(n):
                if nums[i] == target:
                    distance = min(abs(i - q), n - abs(i - q))
                    min_distance = min(min_distance, distance)
            
            if min_distance == float('inf'):
                answer.append(-1)
            else:
                answer.append(min_distance)
        
        return answer