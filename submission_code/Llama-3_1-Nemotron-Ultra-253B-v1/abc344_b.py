import sys

nums = list(map(int, sys.stdin.read().splitlines()))
for num in reversed(nums):
    print(num)