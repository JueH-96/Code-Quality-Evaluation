# YOUR CODE HERE
N = int(input())
S = input().strip()

def is_11_22_string(S):
    length = len(S)
    if length % 2 == 0:
        return False

    mid = (length + 1) // 2

    if S[mid - 1] != '/':
        return False

    for i in range(mid - 1):
        if S[i] != '1':
            return False

    for i in range(mid, length):
        if S[i] != '2':
            return False

    return True

if is_11_22_string(S):
    print("Yes")
else:
    print("No")