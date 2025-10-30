class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + (nums[i] > nums[max(0, i - 1)] and nums[i] > nums[min(n - 1, i + 1)])
        
        answer = []
        for query in queries:
            if query[0] == 1:
                l, r = query[1], query[2]
                answer.append(prefix_sum[r + 1] - prefix_sum[l])
            else:
                index, val = query[1], query[2]
                nums[index] = val
                prefix_sum = [0] * (n + 1)
                for i in range(n):
                    prefix_sum[i + 1] = prefix_sum[i] + (nums[i] > nums[max(0, i - 1)] and nums[i] > nums[min(n - 1, i + 1)])
        return answer