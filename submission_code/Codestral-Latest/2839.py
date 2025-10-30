class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # Combine nums1 and nums2 into pairs and sort them by nums1 in descending order
        combined = sorted(zip(nums1, nums2), reverse=True)

        # Sort queries by x in descending order
        queries = sorted(enumerate(queries), key=lambda x: x[1][0], reverse=True)

        # Use a list to store the results
        results = [-1] * len(queries)

        # Use a stack to keep track of the maximum sums
        stack = []

        # Iterate through the combined pairs
        for num1, num2 in combined:
            # While the stack is not empty and the current num1 is greater than or equal to the x of the query at the top of the stack
            while stack and stack[-1][1][0] <= num1:
                # Pop the top of the stack
                index, (x, y) = stack.pop()
                # If num2 is greater than or equal to y, update the result for the query
                if num2 >= y:
                    results[index] = num1 + num2

            # Push the current pair onto the stack
            stack.append((None, (num1, num2)))

        # Return the results
        return [results[query[0]] for query in queries]