class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = -1
        for i in range(1 << (n - 1)):
            temp = []
            cost = 0
            cur = []
            for j in range(n):
                cur.append(nums[j])
            
            for j in range(n - 1):
                if (i >> j) & 1:
                    cost += 1
                    new_cur = []
                    for l in range(len(cur) - 1):
                        if l == j:
                            new_cur.append(cur[l] & cur[l+1])
                        elif l < j:
                            new_cur.append(cur[l])
                    if j < len(cur) -1:
                        for l in range(j + 2, len(cur)):
                            new_cur.append(cur[l])
                    cur = new_cur
            if cost <= k:
                or_val = 0
                for x in cur:
                    or_val |= x
                if ans == -1 or or_val < ans:
                    ans = or_val
        return ans