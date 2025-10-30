nums = list(map(int, input().split()))

# Check if all numbers are between 100 and 675, inclusive
all_in_range = all(100 <= num <= 675 for num in nums)

# Check if all numbers are multiples of 25
all_multiples_of_25 = all(num % 25 == 0 for num in nums)

# Check if the sequence is non-decreasing
non_decreasing = all(nums[i] <= nums[i+1] for i in range(7))

if all_in_range and all_multiples_of_25 and non_decreasing:
    print("Yes")
else:
    print("No")