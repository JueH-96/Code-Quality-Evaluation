class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        answer = [0] * n
        
        # We will use a list of tuples to keep track of values and their indices
        indexed_nums1 = list(enumerate(nums1))
        
        # Sort nums1 with their indices to efficiently find the required indices for each i
        indexed_nums1.sort(key=lambda x: x[1])  # Sort by the value in nums1
        
        # This will hold the sorted nums2 values corresponding to the sorted nums1
        sorted_nums2 = [nums2[idx] for idx, _ in indexed_nums1]
        
        # Fenwick Tree / Binary Indexed Tree for prefix maximum sums
        class FenwickTree:
            def __init__(self, size):
                self.size = size
                self.tree = [0] * (size + 1)
            
            def update(self, index, value):
                while index <= self.size:
                    self.tree[index] += value
                    index += index & -index
            
            def query(self, index):
                sum = 0
                while index > 0:
                    sum += self.tree[index]
                    index -= index & -index
                return sum
        
        # Initialize the Fenwick Tree
        fenwick = FenwickTree(n)
        
        # We need to process the queries in reverse order to use the Fenwick Tree efficiently
        # We will store the queries and process them after setting up the initial conditions
        queries = [[] for _ in range(n)]
        for i in range(n):
            queries[indexed_nums1[i][0]].append(i)
        
        # To keep track of the largest k elements seen so far
        import heapq
        max_heap = []
        
        # Process each element in the sorted order
        for i in range(n):
            # Add current nums2 value to the max heap
            heapq.heappush(max_heap, sorted_nums2[i])
            # If heap exceeds size k, remove the smallest element
            if len(max_heap) > k:
                heapq.heappop(max_heap)
            
            # Update the Fenwick Tree with the current sum of the largest k elements
            if i > 0:
                fenwick.update(i, sorted_nums2[i] - sorted_nums2[i-1])
            
            # Process all queries for this index
            for q in queries[i]:
                # The sum of the largest k elements up to index i
                if max_heap:
                    answer[q] = sum(max_heap)
        
        return answer