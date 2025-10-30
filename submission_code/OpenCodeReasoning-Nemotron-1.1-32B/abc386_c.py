k = int(input().strip())
S = input().strip()
T = input().strip()

if S == T:
	print("Yes")
else:
	n = len(S)
	m = len(T)
	if abs(n - m) > 1:
		print("No")
	else:
		if n == m:
			count = 0
			for i in range(n):
				if S[i] != T[i]:
					count += 1
					if count > 1:
						break
			print("Yes" if count == 1 else "No")
		elif n == m - 1:
			i, j = 0, 0
			skipped = 0
			while i < n and j < m:
				if S[i] == T[j]:
					i += 1
					j += 1
				else:
					if skipped == 0:
						skipped = 1
						j += 1
					else:
						break
			if i == n and (j == m or (j == m-1 and skipped == 0)):
				print("Yes")
			else:
				print("No")
		else:
			i, j = 0, 0
			skipped = 0
			while i < n and j < m:
				if S[i] == T[j]:
					i += 1
					j += 1
				else:
					if skipped == 0:
						skipped = 1
						i += 1
					else:
						break
			if j == m and (i == n or (i == n-1 and skipped == 0)):
				print("Yes")
			else:
				print("No")