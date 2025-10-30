N = int(input())
S = input()
T = input()

ans = 'Yes'
for s, t in zip(S, T):
    if s == t:
        continue
    elif (s, t) in [('1', 'l'), ('l', '1'), ('0', 'o'), ('o', '0')]:
        continue
    else:
        ans = 'No'
        break

print(ans)