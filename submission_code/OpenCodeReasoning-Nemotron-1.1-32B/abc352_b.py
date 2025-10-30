def main():
	s = input().strip()
	t = input().strip()
	
	n = len(s)
	m = len(t)
	i = n - 1
	j = m - 1
	res = []
	
	while i >= 0 and j >= 0:
		if t[j] == s[i]:
			res.append(j + 1)
			i -= 1
			j -= 1
		else:
			j -= 1
			
	res.reverse()
	print(" ".join(map(str, res)))

if __name__ == '__main__':
	main()