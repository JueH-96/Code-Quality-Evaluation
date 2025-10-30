nums = []
while True:
    try:
        num = int(input())
        nums.append(num)
        if num == 0:
            break
    except EOFError:
        break

for i in range(len(nums) - 1, -1, -1):
    print(nums[i])