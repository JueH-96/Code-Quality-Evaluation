class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Create a dictionary to store the count of each element in the array
        count = {}
        for i in range(n):
            if nums[i] in count:
                count[nums[i]] += 1
            else:
                count[nums[i]] = 1
        
        # Sort the count dictionary by the count of each element
        sorted_count = sorted(count.items(), key=lambda x: x[1], reverse=True)
        
        # Initialize the minimum number of changes required
        min_changes = 0
        
        # Iterate through the sorted count dictionary
        for i in range(len(sorted_count)):
            # Get the current element and its count
            curr_elem, curr_count = sorted_count[i]
            
            # Check if the current element satisfies the condition
            if abs(curr_elem - (n - 1 - curr_elem)) <= k:
                continue
            
            # Find the minimum number of changes required to make the current element satisfy the condition
            min_changes += n - curr_count
            
            # Update the count dictionary with the new elements
            for j in range(k + 1):
                new_elem = curr_elem + j
                if new_elem in count:
                    count[new_elem] += n - curr_count
                else:
                    count[new_elem] = n - curr_count
        
        return min_changes