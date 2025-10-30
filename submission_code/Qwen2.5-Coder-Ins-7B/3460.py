class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        from sortedcontainers import SortedList
        
        def count_inversions(arr):
            inversions = 0
            sorted_list = SortedList()
            for num in arr:
                inversions += len(sorted_list) - sorted_list.bisect_left(num)
                sorted_list.add(num)
            return inversions
        
        def permute(nums, end, cnt):
            if end == n - 1:
                return 1 if count_inversions(nums[:end+1]) == cnt else 0
            
            total = 0
            for i in range(end + 1):
                nums[end], nums[i] = nums[i], nums[end]
                if count_inversions(nums[:end+1]) <= cnt:
                    total = (total + permute(nums, end + 1, cnt)) % MOD
                nums[end], nums[i] = nums[i], nums[end]
            return total
        
        nums = list(range(n))
        return permute(nums, 0, 0)