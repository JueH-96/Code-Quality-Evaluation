import sys
data = sys.stdin.read().split()
S = data[0]
T = data[1]
min_len = min(len(S), len(T))
diff_found = False
for i in range(min_len):
    if S[i] != T[i]:
        print(i + 1)
        diff_found = True
        break
if not diff_found:
    if len(S) == len(T):
        print(0)
    else:
        print(min_len + 1)