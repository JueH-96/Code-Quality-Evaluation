from itertools import permutations
import math

def is_square(n):
    if n < 0:
        return False
    root = int(math.sqrt(n))
    return root * root == n

N = int(input())
S = input()

# Get all permutations and convert to numbers
nums = set()
for p in permutations(S):
    # Skip numbers starting with 0 unless single digit
    if p[0] == '0' and len(p) > 1:
        continue
    num = int(''.join(p))
    nums.add(num)

# Count squares
count = 0
for num in nums:
    if is_square(num):
        count += 1

print(count)