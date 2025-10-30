class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        import heapq, collections
        
        n = len(nums)
        m = n - x + 1
        
        # Data structures for sliding window median + sum of absolute deviations
        small = []  # max-heap (store -val)
        large = []  # min-heap (store val)
        del_small = collections.Counter()
        del_large = collections.Counter()
        size_small = size_large = 0
        sum_small = sum_large = 0
        
        def clean(heap):
            # lazily remove elements marked for deletion at the top of heap
            nonlocal small, large, del_small, del_large
            if heap is small:
                while small and del_small.get(-small[0], 0) > 0:
                    v = -heapq.heappop(small)
                    del_small[v] -= 1
            else:
                while large and del_large.get(large[0], 0) > 0:
                    v = heapq.heappop(large)
                    del_large[v] -= 1
        
        def rebalance():
            # maintain size_small >= size_large and size_small <= size_large + 1
            nonlocal size_small, size_large, sum_small, sum_large
            if size_small > size_large + 1:
                clean(small)
                v = -heapq.heappop(small)
                size_small -= 1
                sum_small -= v
                heapq.heappush(large, v)
                size_large += 1
                sum_large += v
                clean(small)
            elif size_small < size_large:
                clean(large)
                v = heapq.heappop(large)
                size_large -= 1
                sum_large -= v
                heapq.heappush(small, -v)
                size_small += 1
                sum_small += v
                clean(large)
        
        def add(num):
            nonlocal size_small, size_large, sum_small, sum_large
            if not small or num <= -small[0]:
                heapq.heappush(small, -num)
                size_small += 1
                sum_small += num
            else:
                heapq.heappush(large, num)
                size_large += 1
                sum_large += num
            rebalance()
        
        def remove(num):
            nonlocal size_small, size_large, sum_small, sum_large
            clean(small)
            clean(large)
            med = -small[0]
            if num <= med:
                del_small[num] += 1
                size_small -= 1
                sum_small -= num
                if small and num == -small[0]:
                    clean(small)
            else:
                del_large[num] += 1
                size_large -= 1
                sum_large -= num
                if large and num == large[0]:
                    clean(large)
            rebalance()
        
        def getCost():
            clean(small)
            clean(large)
            med = -small[0]
            # sum |val - med| = med*size_small - sum_small + sum_large - med*size_large
            return med * size_small - sum_small + sum_large - med * size_large
        
        # Precompute the cost of making each length-x subarray constant
        C = [0] * (m + 1)
        # build first window
        for i in range(x):
            add(nums[i])
        C[1] = getCost()
        # slide the window
        for i in range(x, n):
            add(nums[i])
            remove(nums[i - x])
            # window starting at i-x+1 has index (i-x+1)+1 = i-x+2 in C
            C[i - x + 2] = getCost()
        
        # DP: dp[t][i] = min cost to pick t non-overlapping windows among first i windows
        INF = 10**18
        prev = [0] * (m + 1)  # dp[0][*] = 0
        for t in range(1, k + 1):
            cur = [INF] * (m + 1)
            for i in range(1, m + 1):
                # skip window i
                if cur[i - 1] < cur[i]:
                    cur[i] = cur[i - 1]
                # take window i, previous window must end <= start_i - x
                j = i - x
                if j < 0:
                    j = 0
                cost = prev[j] + C[i]
                if cost < cur[i]:
                    cur[i] = cost
            prev = cur
        
        return prev[m]