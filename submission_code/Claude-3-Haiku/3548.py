class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        count = 0
        for num in range(10**(n-1), 10**n):
            if num % k == 0 and str(num) == str(num)[::-1]:
                count += 1
        return count