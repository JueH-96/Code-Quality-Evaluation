import itertools
import operator
from functools import reduce

n, k = map(int, input().split())
a = list(map(int, input().split()))
max_xor = 0

for combo in itertools.combinations(a, k):
    current_xor = reduce(operator.xor, combo, 0)
    if current_xor > max_xor:
        max_xor = current_xor

print(max_xor)