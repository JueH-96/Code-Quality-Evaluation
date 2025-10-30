from collections import Counter
from math import factorial

def count_palindromes(s, k):
    n = len(s)
    counter = Counter(s)
    odd_count = sum(v % 2 for v in counter.values())
    if odd_count > k:
        return 0
    else:
        return factorial(n // 2)

def solve():
    n, k = map(int, input().split())
    s = input()
    print(count_palindromes(s, k))

solve()