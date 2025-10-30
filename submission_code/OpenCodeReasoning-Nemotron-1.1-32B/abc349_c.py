def has_subsequence_2(s, c1, c2):
	i = s.find(c1)
	if i == -1:
		return False
	j = s.find(c2, i + 1)
	return j != -1

def has_subsequence_3(s, c1, c2, c3):
	i = s.find(c1)
	if i == -1:
		return False
	j = s.find(c2, i + 1)
	if j == -1:
		return False
	k = s.find(c3, j + 1)
	return k != -1

def main():
	S = input().strip()
	T = input().strip()
	
	if T[2] != 'X':
		if has_subsequence_3(S, T[0].lower(), T[1].lower(), T[2].lower()):
			print("Yes")
		else:
			print("No")
	else:
		if has_subsequence_2(S, T[0].lower(), T[1].lower()):
			print("Yes")
		elif has_subsequence_3(S, T[0].lower(), T[1].lower(), 'x'):
			print("Yes")
		else:
			print("No")

if __name__ == '__main__':
	main()