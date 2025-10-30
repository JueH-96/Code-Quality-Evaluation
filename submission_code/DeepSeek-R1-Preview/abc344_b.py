nums = []
while True:
    line = input().strip()
    num = int(line)
    nums.append(num)
    if num == 0:
        break

for num in reversed(nums):
    print(num)