S = input().strip()
T = input().strip()

def is_subsequence(target, s):
    ptr = 0
    n = len(target)
    for c in s:
        if ptr < n and c.upper() == target[ptr]:
            ptr += 1
    return ptr == n

case1 = is_subsequence(T, S)
if case1:
    print("Yes")
else:
    if T[2] == 'X' and is_subsequence(T[:2], S):
        print("Yes")
    else:
        print("No")