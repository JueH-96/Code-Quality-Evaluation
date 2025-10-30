class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count_digits(n):
            return len(str(n))

        def count_odd_even(n):
            odd = sum(1 for ch in str(n) if int(ch) % 2 == 1)
            even = sum(1 for ch in str(n) if int(ch) % 2 == 0)
            return odd, even

        res = 0
        for i in range(low, high+1):
            if i % k == 0:
                odd, even = count_odd_even(i)
                if odd == even:
                    res += 1
        return res