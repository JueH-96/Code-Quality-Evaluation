n = int(input())
nums = list(map(int, input().split()))

if len(nums) != n:
    print("No")
else:
    first = nums[0]
    all_same = True
    for num in nums:
        if num != first:
            all_same = False
            break
    print("Yes" if all_same else "No")