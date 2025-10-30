import sys

nums = []
for line in sys.stdin:
    stripped = line.strip()
    if stripped:
        nums.append(int(stripped))

for num in reversed(nums):
    print(num)