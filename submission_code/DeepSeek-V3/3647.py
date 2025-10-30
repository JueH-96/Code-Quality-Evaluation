class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        
        # Create a difference array to represent the total decrements each index can receive
        diff = [0] * (n + 1)
        
        # For each query, increment the start and decrement the end+1
        for l, r in queries:
            diff[l] += 1
            if r + 1 < n:
                diff[r+1] -= 1
        
        # Compute the prefix sum to get the total decrements each index can receive
        total = [0] * n
        total[0] = diff[0]
        for i in range(1, n):
            total[i] = total[i-1] + diff[i]
        
        # Check if the total decrements are at least the nums values
        for i in range(n):
            if total[i] < nums[i]:
                return -1
        
        # Now, we need to find the maximum number of queries that can be removed
        # We can remove a query if removing it does not make any total[i] < nums[i]
        # To do this, we need to find the minimal set of queries that cover all the necessary decrements
        
        # We can model this as a problem of finding the minimal set of intervals that cover all the required decrements
        # But since the problem is to maximize the number of queries to remove, we need to find the maximum number of queries that can be removed without affecting the total decrements
        
        # One way is to sort the queries and try to remove them one by one, checking if the remaining queries still satisfy the total decrements
        
        # However, with n and m up to 1e5, we need a more efficient approach
        
        # Instead, we can precompute the necessary decrements and see which queries are redundant
        
        # Let's create a list of the required decrements for each index
        required = [nums[i] for i in range(n)]
        
        # Now, we need to find the minimal set of queries that can cover all the required decrements
        # This is similar to the set cover problem, but since the intervals are overlapping, we can use a greedy approach
        
        # Sort the queries by their end points
        queries_sorted = sorted(queries, key=lambda x: x[1])
        
        # Initialize a list to keep track of the remaining required decrements
        remaining = required.copy()
        
        # Initialize a list to keep track of the used queries
        used = [False] * m
        
        # Iterate through the sorted queries and try to use them to cover the required decrements
        for idx, (l, r) in enumerate(queries_sorted):
            # Check if the query can contribute to any remaining required decrements
            can_use = False
            for i in range(l, r+1):
                if remaining[i] > 0:
                    can_use = True
                    break
            if can_use:
                used[idx] = True
                # Decrement the remaining required decrements
                for i in range(l, r+1):
                    if remaining[i] > 0:
                        remaining[i] -= 1
        
        # Now, count the number of queries that are not used
        not_used = used.count(False)
        
        return not_used