def main():
	K = int(input().strip())
	S = input().strip()
	T = input().strip()
	
	n = len(S)
	m = len(T)
	
	if S == T:
		print("Yes")
	elif n == m:
		diff = 0
		for i in range(n):
			if S[i] != T[i]:
				diff += 1
				if diff > 1:
					break
		if diff == 1:
			print("Yes")
		else:
			print("No")
	elif n == m + 1:
		i = 0
		while i < m and S[i] == T[i]:
			i += 1
		if i == m:
			print("Yes")
		else:
			if S[i+1:] == T[i:]:
				print("Yes")
			else:
				print("No")
	elif m == n + 1:
		i = 0
		while i < n and S[i] == T[i]:
			i += 1
		if i == n:
			print("Yes")
		else:
			if T[i+1:] == S[i:]:
				print("Yes")
			else:
				print("No")
	else:
		print("No")

if __name__ == '__main__':
	main()