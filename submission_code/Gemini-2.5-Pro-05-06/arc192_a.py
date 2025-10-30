import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    num_zeros = 0
    for x in A:
        if x == 0:
            num_zeros += 1

    if num_zeros == 0:
        print("Yes")
        return

    if N == 3:
        if num_zeros == N: # A = (0,0,0)
            print("No")
        else: # At least one 1, e.g. (0,0,1), (0,1,0), (1,0,0), (0,1,1), etc.
            print("Yes")
    else: # N != 3 (so N >= 4, given problem constraint N >= 3)
        if num_zeros == N: # All A_i are 0
            # The construction of S to enable operations on all even/odd positions
            # is possible if M (number of operations) is even.
            # M = N/2 if N is even. M = (N+1)/2 if N is odd (not N=3).
            # M is odd if N % 4 == 1 or N % 4 == 2.
            if N % 4 == 0 or N % 4 == 3:
                print("Yes")
            else: # N % 4 == 1 or N % 4 == 2
                print("No")
        else: # At least one A_i is 1
            # If there's any A_i = 1, we can "break" the cycle of S character dependencies
            # making S construction always possible.
            print("Yes")

solve()