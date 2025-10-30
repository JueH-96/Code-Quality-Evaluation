# YOUR CODE HERE
def is_321_like_number(N):
    digits = list(map(int, str(N)))
    for i in range(len(digits) - 1):
        if digits[i] <= digits[i+1]:
            return False
    return True

N = int(input())
if is_321_like_number(N):
    print("Yes")
else:
    print("No")