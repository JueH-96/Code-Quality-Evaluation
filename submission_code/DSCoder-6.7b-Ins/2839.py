class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # Sort the queries in descending order based on x_i
        queries = [(x_i, y_i, i) for x_i, y_i, i in queries]
        queries.sort(reverse=True)

        # Sort the pairs in descending order based on nums1[i]
        pairs = sorted((nums1[i], nums2[i], i) for i in range(len(nums1)))
        pairs.sort(reverse=True)

        # Initialize an empty heap
        heap = []

        # Initialize an empty list to store the results
        res = [-1] * len(queries)

        # Iterate over the pairs
        for num1, num2, idx in pairs:
            # While the heap is not empty and the maximum element in the heap is smaller than num2, pop the maximum element from the heap
            while heap and heap[0][0] < num2:
                heapq.heappop(heap)

            # If the heap is not empty, update the result for the current query
            if heap:
                res[heap[0][1]] = max(res[heap[0][1]], num1 + num2)

            # Push the pair into the heap
            heapq.heappush(heap, (num2, idx))

        # Iterate over the queries
        for x_i, y_i, i in queries:
            # If the maximum element in the heap is smaller than y_i, update the result for the current query
            if heap and heap[0][0] < y_i:
                res[i] = -1

        return res