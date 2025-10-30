k = int(input())
nums = []
for mask in range(1, 1024):
    if mask == 1:
        continue
    digits = []
    for i in range(10):
        if mask & (1 << i):
            digits.append(i)
    digits_sorted = sorted(digits, reverse=True)
    num = int(''.join(map(str, digits_sorted)))
    nums.append(num)
nums.sort()
print(nums[k-1])