import sys

# Read input
A, B, C = map(int, sys.stdin.read().split())

sum_total = A + B + C

# Check for two-group case (sum is even and one element is half the sum)
if sum_total % 2 == 0:
    target_half = sum_total // 2
    if A == target_half or B == target_half or C == target_half:
        print("Yes")
        sys.exit()

# Check for three-group case (sum divisible by 3 and all elements equal)
if sum_total % 3 == 0:
    if A == B == C:
        print("Yes")
        sys.exit()

# If none of the conditions are met
print("No")