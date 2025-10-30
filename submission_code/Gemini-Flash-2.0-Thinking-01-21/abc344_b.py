# YOUR CODE HERE
nums = []
while True:
    line = input()
    num = int(line)
    nums.append(num)
    if num == 0:
        break

nums.reverse()
for num in nums:
    print(num)