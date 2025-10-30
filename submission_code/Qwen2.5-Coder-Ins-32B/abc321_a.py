# YOUR CODE HERE
import sys

def is_321_like_number(N):
    digits = list(str(N))
    for i in range(len(digits) - 1):
        if digits[i] <= digits[i + 1]:
            return "No"
    return "Yes"

N = int(sys.stdin.read().strip())
print(is_321_like_number(N))