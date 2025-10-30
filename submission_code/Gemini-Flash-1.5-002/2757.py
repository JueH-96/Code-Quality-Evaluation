class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def digit_sum(n):
            s = 0
            while n:
                s += n % 10
                n //= 10
            return s

        def count_range(n, min_sum, max_sum):
            count = 0
            q = [(0, 0)]  # (current_number, current_sum)
            while q:
                curr_num, curr_sum = q.pop(0)
                if curr_num > n:
                    continue
                if curr_num > 0 and min_sum <= curr_sum <= max_sum:
                    count +=1

                for i in range(10):
                    next_num = curr_num * 10 + i
                    if next_num <= n:
                        q.append((next_num, curr_sum + i))
            return count

        num1_int = int(num1)
        num2_int = int(num2)

        ans = count_range(num2_int,min_sum, max_sum) - count_range(num1_int -1, min_sum, max_sum)

        return ans % MOD