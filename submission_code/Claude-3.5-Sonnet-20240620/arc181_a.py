# YOUR CODE HERE
def solve_case(N, P):
    operations = 0
    left_sorted = 0
    right_sorted = N + 1

    while left_sorted < right_sorted - 1:
        if P[left_sorted] == left_sorted + 1:
            left_sorted += 1
        elif P[right_sorted - 2] == right_sorted - 1:
            right_sorted -= 1
        else:
            operations += 1
            left_sorted = 0
            right_sorted = N + 1

    return operations

T = int(input())
for _ in range(T):
    N = int(input())
    P = list(map(int, input().split()))
    print(solve_case(N, P))