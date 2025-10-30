class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        if k % 2 == 0:
            even = [i for i in range(0, 10, 2) if i % k == 0]
            odd = [i for i in range(1, 10, 2) if i % k == 0]
            if n % 2 == 0:
                res = sum([comb(even, n // 2) + comb(odd, n // 2) for odd in odd for even in even])
            else:
                res = sum([comb(even, n // 2) * comb(odd, n // 2 + 1) + comb(odd, n // 2) * comb(even, n // 2 + 1) for odd in odd for even in even])
        else:
            even = [i for i in range(0, 10, 2) if i % k == 0]
            odd = [i for i in range(1, 10, 2) if i % k == 0]
            if n % 2 == 0:
                res = sum([comb(even, n // 2) * comb(odd, n // 2) for odd in odd for even in even])
            else:
                res = sum([comb(even, n // 2) * comb(odd, n // 2 + 1) for odd in odd for even in even])
        return res

def comb(arr, n):
    if n == 0:
        return 1
    if len(arr) == 1:
        return 1 if arr[0] != 0 else 0
    if n == 1:
        return len(arr)
    res = 0
    for i in range(len(arr)):
        if arr[i] != 0:
            res += comb(arr[i + 1:], n - 1)
    return res