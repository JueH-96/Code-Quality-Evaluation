nums = []
while True:
    try:
        num = int(input())
        nums.append(num)
        if num == 0:
            break
    except EOFError:
        break

for num in reversed(nums):
    print(num)