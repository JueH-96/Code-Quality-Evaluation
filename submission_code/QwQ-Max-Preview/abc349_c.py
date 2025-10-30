s = input().strip()
t = input().strip()

def check_subsequence(target, s_str):
    ptr = 0
    n = len(target)
    for c in s_str:
        if ptr >= n:
            break
        if c == target[ptr].lower():
            ptr += 1
    return ptr == n

# Check case 1: T is a 3-character subsequence
case1 = check_subsequence(t, s)
if case1:
    print("Yes")
else:
    # Check case 2: T ends with X and first two are a subsequence
    if t[2] == 'X':
        case2 = check_subsequence(t[:2], s)
        print("Yes" if case2 else "No")
    else:
        print("No")