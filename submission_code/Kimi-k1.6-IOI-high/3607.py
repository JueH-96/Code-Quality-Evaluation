class Solution:
    _spf = None  # Class variable to hold the smallest prime factor sieve

    @classmethod
    def _initialize_spf(cls):
        max_num = 10**6
        spf = list(range(max_num + 1))
        for i in range(2, int(max_num**0.5) + 1):
            if spf[i] == i:
                for j in range(i * i, max_num + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        cls._spf = spf

    def __init__(self):
        if Solution._spf is None:
            Solution._initialize_spf()

    def minOperations(self, nums: List[int]) -> int:
        spf = Solution._spf
        prev = float('inf')
        res = 0

        for x in reversed(nums):
            if x == 1:
                prev = 1
            else:
                p = spf[x]
                if p == x:  # Prime number
                    if x > prev:
                        return -1
                    prev = x
                else:  # Composite number
                    if x <= prev:
                        prev = x
                    elif p <= prev:
                        res += 1
                        prev = p
                    else:
                        return -1
        return res