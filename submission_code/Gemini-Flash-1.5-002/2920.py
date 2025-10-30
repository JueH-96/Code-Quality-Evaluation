class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        if all(x == nums[0] for x in nums):
            return 0

        def check(seconds):
            if seconds == 0:
                return all(x == nums[0] for x in nums)

            
            possible_values = set(nums)
            
            for _ in range(seconds):
                new_nums = []
                for i in range(n):
                    options = [nums[i], nums[(i - 1 + n) % n], nums[(i + 1) % n]]
                    new_nums.append(options[0])

                nums = new_nums
                possible_values = set(nums)
                if len(possible_values) ==1:
                    return True

            return len(possible_values) == 1


        left, right = 0, n
        ans = n
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        return ans