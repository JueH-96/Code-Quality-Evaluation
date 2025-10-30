class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        d = {}
        for i, v in enumerate(nums):
            if v not in d:
                d[v] = [i]
            else:
                d[v].append(i)
        ans = inf
        for lst in d.values():
            new_lst = lst + [lst[0] + len(nums)]
            ans = min(ans, max((new_lst[i + 1] - new_lst[i]) // 2 for i in range(len(lst))))
        return ans