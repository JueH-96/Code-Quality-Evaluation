def is_subsequence(s, pattern):
	n = len(pattern)
	idx = 0
	for char in s:
		if idx == n:
			break
		if char.upper() == pattern[idx]:
			idx += 1
	return idx == n

def main():
	S = input().strip()
	T = input().strip()
	
	if is_subsequence(S, T):
		print("Yes")
	elif T[2] == 'X' and is_subsequence(S, T[:2]):
		print("Yes")
	else:
		print("No")

if __name__ == "__main__":
	main()