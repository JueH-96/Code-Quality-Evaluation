from typing import List
import heapq

class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        """
        Pick at most k elements from the matrix maximizing their sum, with the
        restriction that row i contributes at most limits[i] elements.
        All grid entries are non-negative, so using fewer than the allowed
        amount can never increase the sum – we will finally take exactly
        min(k, total_available) elements.
        """
        if k == 0:                 # nothing can be taken
            return 0
        
        # The largest k elements among all rows will be tracked in a min-heap
        # of size at most k.
        largest = []               # min-heap
        heapq.heapify(largest)

        for row, lim in zip(grid, limits):
            if lim == 0:
                continue           # this row cannot contribute anything

            # grab the 'lim' largest numbers of the current row
            # nlargest runs in O(len(row) * log(lim)) time
            for val in heapq.nlargest(lim, row):
                if len(largest) < k:
                    heapq.heappush(largest, val)
                else:
                    # keep only the current k biggest values seen so far
                    if val > largest[0]:
                        heapq.heapreplace(largest, val)

        # when k exceeds the total number of collected elements
        # we simply sum everything that is available
        return sum(largest)