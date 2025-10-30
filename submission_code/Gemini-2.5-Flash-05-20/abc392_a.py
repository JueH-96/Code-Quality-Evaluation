import sys

def solve():
    # Read the three integers from standard input
    A = list(map(int, sys.stdin.readline().split()))

    # A contains A_1, A_2, A_3. Let's refer to them as a, b, c for simplicity.
    a, b, c = A[0], A[1], A[2]

    # We need to check if any permutation (B_1, B_2, B_3) of (a, b, c)
    # satisfies the condition B_1 * B_2 = B_3.

    # There are 3 distinct ways to pick two elements for B_1, B_2,
    # and the remaining one for B_3:
    # 1. B_1 = a, B_2 = b, B_3 = c  (or B_1 = b, B_2 = a, B_3 = c due to commutativity)
    # 2. B_1 = a, B_2 = c, B_3 = b  (or B_1 = c, B_2 = a, B_3 = b)
    # 3. B_1 = b, B_2 = c, B_3 = a  (or B_1 = c, B_2 = b, B_3 = a)

    # Check these three conditions:
    if a * b == c:
        print("Yes")
        return
    if a * c == b:
        print("Yes")
        return
    if b * c == a:
        print("Yes")
        return

    # If none of the above conditions are met, then it's not possible.
    print("No")

# Call the solve function to execute the program
if __name__ == '__main__':
    solve()