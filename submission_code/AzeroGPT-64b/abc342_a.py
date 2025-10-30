S = input()
for i, s in enumerate(S):
	if S.count(s) == 1:
		break

print(i+1)