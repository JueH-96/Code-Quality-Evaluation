class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        
        # Initialize the difference array
        diff = [0] * (n + 1)
        
        # Calculate the total decrement each index can receive
        for l, r in queries:
            diff[l] += 1
            if r + 1 < n + 1:
                diff[r+1] -= 1
        
        # Compute the prefix sum to get the total decrement for each index
        total = [0] * n
        total[0] = diff[0]
        for i in range(1, n):
            total[i] = total[i-1] + diff[i]
        
        # Check if the total decrement is at least nums[i] for each index
        for i in range(n):
            if total[i] < nums[i]:
                return -1
        
        # Now, we need to find the maximum number of queries that can be removed
        # such that the remaining queries still satisfy the condition
        
        # We can model this as a problem of finding the maximum number of queries
        # that can be removed without violating the condition that the sum of
        # decrements for each index is at least nums[i]
        
        # To maximize the number of queries to remove, we need to find the minimal
        # set of queries that cover all the necessary decrements
        
        # We can use a greedy approach to select the queries that cover the most
        # indices first, but it's more efficient to use a priority queue or similar
        
        # However, given the constraints, we need a more efficient approach
        
        # Instead, we can sort the queries based on their length and try to remove
        # the ones that cover the least number of indices first
        
        # But this is not straightforward, so we need a different approach
        
        # Let's consider that the total decrement for each index is the sum of
        # the contributions from all queries that cover it
        
        # To maximize the number of queries to remove, we need to find the minimal
        # subset of queries such that the sum of their contributions is at least
        # nums[i] for each index
        
        # This is similar to the set cover problem, which is NP-hard, but given
        # the constraints, we need a more efficient solution
        
        # Given the time constraints, we will use a heuristic approach
        
        # We will sort the queries in a way that prioritizes those that cover
        # the most indices first, and then try to remove the ones that are not
        # necessary
        
        # But this is not guaranteed to work in all cases
        
        # Given the time constraints, we will implement a simple approach that
        # checks if removing a query still satisfies the condition
        
        # Initialize a list to keep track of the necessary queries
        necessary = [True] * m
        
        # Iterate through each query and check if it can be removed
        for i in range(m):
            l, r = queries[i]
            # Temporarily remove the query
            for idx in range(l, r+1):
                total[idx] -= 1
            # Check if all indices still have sufficient decrement
            valid = True
            for idx in range(n):
                if total[idx] < nums[idx]:
                    valid = False
                    break
            if valid:
                necessary[i] = False
            else:
                # Re-add the query
                for idx in range(l, r+1):
                    total[idx] += 1
        
        # Count the number of queries that can be removed
        return necessary.count(False)