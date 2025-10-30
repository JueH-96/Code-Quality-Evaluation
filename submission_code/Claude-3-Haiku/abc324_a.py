# Read the input
n = int(input())
nums = list(map(int, input().split()))

# Check if all elements are equal
if len(set(nums)) == 1:
    print("Yes")
else:
    print("No")