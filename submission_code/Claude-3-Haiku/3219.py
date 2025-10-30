class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        
        # Create a list of tuples (value, index) and sort it
        pairs = sorted([(nums[i], i) for i in range(n)])
        
        # Initialize the result array
        result = [0] * n
        
        # Iterate through the sorted pairs and swap the elements if the difference is within the limit
        for i in range(n):
            for j in range(i+1, n):
                if abs(pairs[i][0] - pairs[j][0]) <= limit:
                    result[pairs[i][1]] = pairs[j][0]
                    result[pairs[j][1]] = pairs[i][0]
                    break
            else:
                result[pairs[i][1]] = pairs[i][0]
        
        return result