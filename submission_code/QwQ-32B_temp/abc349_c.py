s = input().strip()
t = input().strip()

def is_subsequence(s, sub):
    it = iter(s)
    return all(c in it for c in sub)

# Check case 1: T is a 3-letter subsequence
case1 = is_subsequence(s, t.lower())

# Check case 2: first two letters of T form a subsequence and third is 'X'
case2 = False
if t[2] == 'X':
    sub_two = t[0:2].lower()
    case2 = is_subsequence(s, sub_two)

if case1 or case2:
    print("Yes")
else:
    print("No")