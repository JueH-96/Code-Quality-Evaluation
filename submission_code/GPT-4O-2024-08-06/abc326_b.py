# YOUR CODE HERE
def find_326_like_number(N):
    # Start checking from N
    for num in range(N, 920):  # 920 is exclusive, so it checks up to 919
        # Convert the number to a string to easily access digits
        str_num = str(num)
        hundreds = int(str_num[0])
        tens = int(str_num[1])
        ones = int(str_num[2])
        
        # Check if the product of hundreds and tens equals the ones digit
        if hundreds * tens == ones:
            return num

# Read input
import sys
input = sys.stdin.read
N = int(input().strip())

# Find and print the smallest 326-like number greater than or equal to N
print(find_326_like_number(N))