# Read input
a1, a2, a3 = map(int, input().split())

# Function to check if a permutation exists where b1 * b2 = b3
def check_multiplication(nums):
    # Try all possible permutations
    # Since we only have 3 numbers, we can manually check all 6 permutations
    permutations = [
        (nums[0], nums[1], nums[2]),
        (nums[0], nums[2], nums[1]),
        (nums[1], nums[0], nums[2]),
        (nums[1], nums[2], nums[0]),
        (nums[2], nums[0], nums[1]),
        (nums[2], nums[1], nums[0])
    ]
    
    # Check each permutation
    for b1, b2, b3 in permutations:
        if b1 * b2 == b3:
            return True
    return False

# Check if any permutation satisfies the condition
if check_multiplication([a1, a2, a3]):
    print("Yes")
else:
    print("No")