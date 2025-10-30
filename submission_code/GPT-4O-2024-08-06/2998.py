class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        def is_symmetric(x):
            s = str(x)
            if len(s) % 2 != 0:
                return False
            n = len(s) // 2
            first_half_sum = sum(int(s[i]) for i in range(n))
            second_half_sum = sum(int(s[i]) for i in range(n, len(s)))
            return first_half_sum == second_half_sum
        
        count = 0
        for num in range(low, high + 1):
            if is_symmetric(num):
                count += 1
        
        return count