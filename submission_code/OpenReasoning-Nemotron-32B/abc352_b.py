def main():
	s = input().strip()
	t = input().strip()
	
	i = 0
	j = 0
	res = []
	n = len(s)
	m = len(t)
	
	while i < n and j < m:
		if s[i] == t[j]:
			res.append(j + 1)
			i += 1
			j += 1
		else:
			j += 1
			
	print(" ".join(map(str, res)))

if __name__ == "__main__":
	main()