class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        import heapq
        marked = [0]*len(nums)
        unmarked = []
        for i, num in enumerate(nums):
            heapq.heappush(unmarked, (num, i))
        res = []
        total = sum(nums)
        for index, k in queries:
            if marked[index] == 0:
                marked[index] = 1
                total -= nums[index]
            count = 0
            while unmarked and count < k:
                num, i = heapq.heappop(unmarked)
                if marked[i] == 0:
                    marked[i] = 1
                    total -= num
                    count += 1
            res.append(total)
        return res