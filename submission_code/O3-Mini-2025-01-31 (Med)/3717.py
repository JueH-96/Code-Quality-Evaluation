from typing import List
import heapq
import math

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        m = n - x + 1  # number of candidate windows
        
        # We'll use two heaps with lazy deletion. left is a max-heap (store negatives), right is a min-heap.
        # They will maintain the current window of length x.
        left = []     # max-heap (store negative values)
        right = []    # min-heap
        sum_left = 0  # sum of values in left (active only)
        sum_right = 0 # sum of values in right (active only)
        nleft = 0     # count of active elements in left
        nright = 0    # count of active elements in right
        
        # dictionaries for lazy deletion (for values that should be removed from that heap)
        delayed_left = {}
        delayed_right = {}
        
        # Helper: prune the top of heap if it is marked for deletion.
        def prune(heap, delayed):
            nonlocal nleft, nright, sum_left, sum_right
            while heap and delayed.get(heap[0][1], 0) > 0:
                # pop the top value from heap (heap element is a tuple (val, origVal)).
                val, orig_val = heapq.heappop(heap)[1:]
                delayed[orig_val] -= 1
                if delayed[orig_val] == 0:
                    del delayed[orig_val]
            # no return

        # Because we want to store both the effective value for ordering and a secondary item,
        # we will store each heap element as a tuple (key, actual_value)
        # For left (max heap), key is -value.
        # For right (min heap), key is value.
        
        # Helper functions to add element v to the heaps.
        def addNum(v):
            nonlocal nleft, nright, sum_left, sum_right
            # if left is empty or v <= current median then push to left, else push to right.
            if nleft == 0:
                heapq.heappush(left, (-v, v))
                nleft += 1
                sum_left += v
            else:
                # get current median (top of left)
                median = left[0][1]
                if v <= median:
                    heapq.heappush(left, (-v, v))
                    nleft += 1
                    sum_left += v
                else:
                    heapq.heappush(right, (v, v))
                    nright += 1
                    sum_right += v
                    
            rebalance()
        
        def removeNum(v):
            nonlocal nleft, nright, sum_left, sum_right
            # Decide from which heap v will be removed.
            median = left[0][1]  # current median from left should be valid (after pruning, if needed)
            if v <= median:
                # remove from left (lazy)
                delayed_left[v] = delayed_left.get(v, 0) + 1
                nleft -= 1
                sum_left -= v
                # if the top of left equals v, then prune immediately.
                if left and left[0][1] == v:
                    prune(left, delayed_left)
            else:
                delayed_right[v] = delayed_right.get(v, 0) + 1
                nright -= 1
                sum_right -= v
                if right and right[0][1] == v:
                    prune(right, delayed_right)
            rebalance()
        
        def rebalance():
            nonlocal nleft, nright, sum_left, sum_right
            # Ensure that nleft == nright or nleft == nright + 1 (so that left has the median)
            if nleft > nright + 1:
                # move one element from left to right
                prune(left, delayed_left)
                if left:
                    _, v = heapq.heappop(left)
                    nleft -= 1
                    sum_left -= v
                    heapq.heappush(right, (v, v))
                    nright += 1
                    sum_right += v
            elif nleft < nright:
                prune(right, delayed_right)
                if right:
                    _, v = heapq.heappop(right)
                    nright -= 1
                    sum_right -= v
                    heapq.heappush(left, (-v, v))
                    nleft += 1
                    sum_left += v

            # After rebalancing, always prune tops if needed.
            if left:
                prune(left, delayed_left)
            if right:
                prune(right, delayed_right)
                
        def getMedian():
            prune(left, delayed_left)
            if left:
                return left[0][1]
            return None

        def getCost():
            # Given median m, the cost to make all numbers equal to m is:
            # cost = (m * (nleft)) - sum_left + sum_right - m*(nright)
            m = getMedian()
            return m * nleft - sum_left + sum_right - m * nright
        
        # Build the first window (indices 0 to x-1)
        for i in range(x):
            addNum(nums[i])
            
        # We'll record cost for the first window starting at index 0.
        costs = [getCost()]
        
        # Slide the window from index 1 to index m-1.
        for i in range(1, m):
            # remove the element leaving the window and add the new element entering.
            removeNum(nums[i-1])
            addNum(nums[i+x-1])
            costs.append(getCost())
        
        # Now we have a list costs of length m where costs[i] is the cost to flatten window starting at i.
        # Next, use DP to choose k non overlapping segments.
        # Let dp[r][i] be the minimal cost to pick r segments from candidates starting at index i.
        INF = math.inf
        # We need dp for r = 0,1,...,k, and positions i = 0,...,m
        dp = [[INF] * (m + 1) for _ in range(k + 1)]
        # Base: dp[0][i] = 0 for all i (zero segments cost 0)
        for i in range(m + 1):
            dp[0][i] = 0
        
        # Fill dp for r from 1 to k.
        # We iterate backwards over positions.
        for r in range(1, k + 1):
            for i in range(m - 1, -1, -1):
                # Option 1: skip candidate i.
                option1 = dp[r][i+1]
                # Option 2: take candidate i (if possible) and then next candidate can only start at i + x.
                j = i + x
                if j <= m:
                    option2 = costs[i] + dp[r-1][j]
                else:
                    option2 = INF
                dp[r][i] = min(option1, option2)
        
        return dp[k][0]