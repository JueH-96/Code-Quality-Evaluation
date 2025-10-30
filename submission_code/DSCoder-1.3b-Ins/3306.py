class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False]*n
        unmarked = [nums[i] for i in range(n) if not marked[i]]
        unmarked.sort()
        res = []
        for q in queries:
            index, k = q[0], q[1]
            marked[index] = True
            count = 0
            for i in range(len(unmarked)):
                if unmarked[i] < unmarked[0]:
                    count += 1
                if count == k:
                    break
            if count < k:
                res.append(0)
            else:
                res.append(unmarked[count-k])
        return res