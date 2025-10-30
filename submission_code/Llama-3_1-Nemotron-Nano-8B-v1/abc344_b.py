import sys

lines = sys.stdin.read().splitlines()
nums = [int(line) for line in lines]
for num in reversed(nums):
    print(num)