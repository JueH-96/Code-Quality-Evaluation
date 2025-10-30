# YOUR CODE HERE
def check_number(N):
    digits = str(N)
    return digits.count('1') == 1 and digits.count('2') == 2 and digits.count('3') == 3

N = int(input())
if check_number(N):
    print("Yes")
else:
    print("No")