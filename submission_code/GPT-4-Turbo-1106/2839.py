class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # Combine nums1 and nums2 into a list of tuples and sort it in descending order based on the sum
        combined = sorted(zip(nums1, nums2), key=lambda x: -(x[0] + x[1]))
        
        # Prepare a list to store the maximum sum for each prefix
        max_sums = []
        current_max = -1
        
        # Iterate over the combined list to fill the max_sums list
        for num1, num2 in combined:
            current_max = max(current_max, num1 + num2)
            max_sums.append(current_max)
        
        # Prepare the answer list
        answer = []
        
        # Iterate over the queries to find the maximum sum for each query
        for x, y in queries:
            # Initialize the maximum sum for the current query as -1
            max_sum = -1
            
            # Binary search to find the rightmost pair that satisfies the constraints
            left, right = 0, len(combined) - 1
            while left <= right:
                mid = (left + right) // 2
                if combined[mid][0] >= x and combined[mid][1] >= y:
                    max_sum = max_sums[mid]
                    left = mid + 1
                else:
                    right = mid - 1
            
            # Append the maximum sum to the answer list
            answer.append(max_sum)
        
        return answer