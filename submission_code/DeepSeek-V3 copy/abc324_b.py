# YOUR CODE HERE
def is_exp_of_2_and_3(N):
    if N < 1:
        return False
    while N % 2 == 0:
        N = N // 2
    while N % 3 == 0:
        N = N // 3
    return N == 1

N = int(input())
if is_exp_of_2_and_3(N):
    print("Yes")
else:
    print("No")