def main():
	n = int(input().strip())
	s = input().strip()
	t = input().strip()
	
	for i in range(n):
		if s[i] == t[i]:
			continue
		if s[i] in '1l' and t[i] in '1l':
			continue
		if s[i] in '0o' and t[i] in '0o':
			continue
		print("No")
		return
	
	print("Yes")

if __name__ == "__main__":
	main()