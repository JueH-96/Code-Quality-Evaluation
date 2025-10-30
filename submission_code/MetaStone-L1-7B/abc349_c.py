def is_subsequence(target, s):
    ptr = 0
    for c in target:
        found = False
        while ptr < len(s):
            if s[ptr] == c:
                found = True
                break
            ptr += 1
        if not found:
            return False
    return True

S = input().strip()
T = input().strip()

case1 = is_subsequence(T, S)
if case1:
    print("Yes")
else:
    if T[2] == 'X':
        first_two = T[:2]
        case2 = is_subsequence(first_two, S)
        if case2:
            print("Yes")
        else:
            print("No")
    else:
        print("No")