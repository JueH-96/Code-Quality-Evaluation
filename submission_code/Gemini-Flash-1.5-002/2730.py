class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            ans |= nums[i]

        q = [(nums,k,ans)]
        visited = {tuple(nums)}
        max_or = ans

        while q:
            curr_nums, curr_k, curr_or = q.pop(0)

            max_or = max(max_or, curr_or)

            for i in range(n):
                next_nums = list(curr_nums)
                if curr_k > 0:
                    next_nums[i] *= 2
                    next_or = 0
                    for x in next_nums:
                        next_or |= x
                    
                    if tuple(next_nums) not in visited and curr_k -1 >=0:
                        visited.add(tuple(next_nums))
                        q.append((next_nums, curr_k - 1, next_or))

        return max_or