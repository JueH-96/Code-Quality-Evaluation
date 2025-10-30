# YOUR CODE HERE
def is_321_like(n):
    digits = str(n)
    for i in range(len(digits) - 1):
        if int(digits[i]) <= int(digits[i+1]):
            return False
    return True

N = int(input())
print("Yes" if is_321_like(N) else "No")