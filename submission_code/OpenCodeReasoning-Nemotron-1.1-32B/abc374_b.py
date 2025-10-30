S = input().strip()
T = input().strip()

n = min(len(S), len(T))
for i in range(n):
	if S[i] != T[i]:
		print(i + 1)
		break
else:
	if len(S) == len(T):
		print(0)
	else:
		print(n + 1)