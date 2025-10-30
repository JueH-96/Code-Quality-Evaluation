class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False] * n
        result = []

        for index, k in queries:
            if not marked[index]:
                marked[index] = True

            unmarked_indices = [i for i in range(n) if not marked[i]]
            
            unmarked_values = [(nums[i], i) for i in unmarked_indices]
            unmarked_values.sort()
            
            count = 0
            for value, idx in unmarked_values:
                if count < k:
                    marked[idx] = True
                    count += 1
                else:
                    break
            
            unmarked_sum = 0
            for i in range(n):
                if not marked[i]:
                    unmarked_sum += nums[i]
            result.append(unmarked_sum)

        return result