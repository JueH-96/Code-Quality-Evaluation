class Solution:
    def __init__(self):
        max_num = 10**6
        self.spf = list(range(max_num + 1))
        for i in range(2, int(max_num**0.5) + 1):
            if self.spf[i] == i:
                for j in range(i*i, max_num + 1, i):
                    if self.spf[j] == j:
                        self.spf[j] = i

    def minOperations(self, nums):
        options = []
        for x in nums:
            if x == 1:
                opts = [(1, 0)]
            else:
                if self.spf[x] == x:
                    opts = [(x, 0)]
                else:
                    reduced = x // self.spf[x]
                    opts = [(x, 0), (reduced, 1)]
                    opts.sort(reverse=True, key=lambda t: t[0])
            options.append(opts)
        
        n = len(nums)
        next_max = float('inf')
        total_ops = 0
        for i in range(n-1, -1, -1):
            current_options = options[i]
            found = False
            for val, ops in current_options:
                if val <= next_max:
                    total_ops += ops
                    next_max = val
                    found = True
                    break
            if not found:
                return -1
        return total_ops