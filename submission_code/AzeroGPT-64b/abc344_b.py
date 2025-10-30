nums = []
num = int(input())
while num != 0:
    nums.append(num)
    num = int(input())

for i in reversed(nums):
    print(i)