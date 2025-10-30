import sys
data = sys.stdin.read().split()
nums = [int(x) for x in data]
nums.reverse()
for num in nums:
    print(num)