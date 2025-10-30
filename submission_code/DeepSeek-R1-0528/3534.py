class Solution:
    def countPairs(self, nums: List[int]) -> int:
        distinct_nums = set(nums)
        cache = {}
        for num in distinct_nums:
            s = str(num)
            n = len(s)
            if n == 1:
                cache[num] = set()
            else:
                res_set = set()
                lst = list(s)
                for i in range(n):
                    for j in range(i+1, n):
                        if lst[i] == lst[j]:
                            continue
                        lst[i], lst[j] = lst[j], lst[i]
                        candidate = int(''.join(lst))
                        res_set.add(candidate)
                        lst[i], lst[j] = lst[j], lst[i]
                cache[num] = res_set
        
        count = 0
        n_len = len(nums)
        for i in range(n_len):
            for j in range(i + 1, n_len):
                x = nums[i]
                y = nums[j]
                if x == y:
                    count += 1
                else:
                    if y in cache[x] or x in cache[y]:
                        count += 1
        return count