# Read input from stdin
S = input().strip()
T = input().strip()

def is_subseq(target, s):
    i = 0  # pointer for target
    j = 0  # pointer for s
    while i < len(target) and j < len(s):
        if s[j] == target[i]:
            i += 1
        j += 1
    return i == len(target)

# Check case 1: T is a subsequence of S converted to uppercase
case1 = T.lower()
if is_subseq(case1, S):
    print("Yes")
else:
    # Check case 2: first two letters of T are a subsequence and third is 'X'
    if T[2] == 'X' and is_subseq(T[:2].lower(), S):
        print("Yes")
    else:
        print("No")