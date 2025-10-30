# YOUR CODE HERE
def is_321_like_number(N):
    digits = [int(d) for d in str(N)]
    return all(digits[i] > digits[i+1] for i in range(len(digits)-1))

N = int(input())
if is_321_like_number(N):
    print("Yes")
else:
    print("No")