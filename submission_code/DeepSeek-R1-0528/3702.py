class Solution:
    def maxLength(self, nums: List[int]) -> int:
        primes = [2, 3, 5, 7]
        n = len(nums)
        exp_list = []
        for num in nums:
            exps = []
            for p in primes:
                cnt = 0
                n_val = num
                while n_val % p == 0:
                    cnt += 1
                    n_val //= p
                exps.append(cnt)
            exp_list.append(exps)
        
        max_len = 0
        for left in range(n):
            min_exp = [float('inf')] * 4
            max_exp = [0] * 4
            total_exp = [0] * 4
            for right in range(left, n):
                valid = True
                for i in range(4):
                    e_val = exp_list[right][i]
                    total_exp[i] += e_val
                    if e_val < min_exp[i]:
                        min_exp[i] = e_val
                    if e_val > max_exp[i]:
                        max_exp[i] = e_val
                for i in range(4):
                    if max_exp[i] > 0 and total_exp[i] != min_exp[i] + max_exp[i]:
                        valid = False
                        break
                if valid:
                    current_length = right - left + 1
                    if current_length > max_len:
                        max_len = current_length
        return max_len