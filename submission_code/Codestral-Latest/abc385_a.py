# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

A = int(data[0])
B = int(data[1])
C = int(data[2])

total_sum = A + B + C

# Check if the total sum is odd
if total_sum % 2 != 0:
    print("No")
else:
    # Check if any single number is equal to half of the total sum
    half_sum = total_sum // 2
    if A == half_sum or B == half_sum or C == half_sum:
        print("Yes")
    else:
        # Check if the sum of any two numbers is equal to half of the total sum
        if (A + B == half_sum) or (A + C == half_sum) or (B + C == half_sum):
            print("Yes")
        else:
            print("No")