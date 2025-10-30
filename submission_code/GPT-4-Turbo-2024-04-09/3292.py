class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        # Convert 1-indexed changeIndices to 0-indexed for easier manipulation with nums
        changeIndices = [idx - 1 for idx in changeIndices]
        
        # Track the number of decrements needed for each index to reach zero
        decrements_needed = nums[:]
        
        # Track whether an index is marked
        marked = [False] * n
        
        # Track the last time an index can be marked based on the decrements needed
        last_possible_mark = [-1] * n
        
        # Calculate the last possible second each index can be marked
        for second in range(m):
            index = changeIndices[second]
            if decrements_needed[index] > 0:
                decrements_needed[index] -= 1
            if decrements_needed[index] == 0:
                last_possible_mark[index] = second
        
        # Check if all indices can be marked and find the latest second needed
        latest_second = -1
        for i in range(n):
            if last_possible_mark[i] == -1:
                return -1  # If any index cannot be marked, return -1
            latest_second = max(latest_second, last_possible_mark[i])
        
        # Return the 1-indexed second (add 1 to the 0-indexed latest_second)
        return latest_second + 1