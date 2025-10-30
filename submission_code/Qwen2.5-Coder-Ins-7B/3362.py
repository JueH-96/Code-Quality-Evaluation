class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        from sortedcontainers import SortedList
        
        n = len(nums)
        left = [0] * n
        right = [0] * n
        last = [-1] * (10**5 + 1)
        
        for i in range(n):
            if last[nums[i]] != -1:
                left[i] = left[last[nums[i]]] + i - last[nums[i]] - 1
            else:
                left[i] = i
            last[nums[i]] = i
        
        last = [-1] * (10**5 + 1)
        for i in range(n-1, -1, -1):
            if last[nums[i]] != -1:
                right[i] = right[last[nums[i]]] + last[nums[i]] - i - 1
            else:
                right[i] = n - 1 - i
            last[nums[i]] = i
        
        uniques = SortedList()
        for i in range(n):
            uniques.add(left[i] + right[i] + 1)
        
        return uniques[n // 2]