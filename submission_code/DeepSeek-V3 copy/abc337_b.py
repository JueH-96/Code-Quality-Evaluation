# YOUR CODE HERE
def is_extended_abc_string(s):
    n = len(s)
    if n == 0:
        return True
    # Find the transition points
    a_end = -1
    b_end = -1
    # Find the end of A's
    for i in range(n):
        if s[i] != 'A':
            a_end = i - 1
            break
    else:
        a_end = n - 1
    # Find the end of B's
    for i in range(a_end + 1, n):
        if s[i] != 'B':
            b_end = i - 1
            break
    else:
        b_end = n - 1
    # Check the remaining part for C's
    for i in range(b_end + 1, n):
        if s[i] != 'C':
            return False
    return True

S = input().strip()
if is_extended_abc_string(S):
    print("Yes")
else:
    print("No")