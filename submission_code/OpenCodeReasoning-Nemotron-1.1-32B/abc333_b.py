def main():
	mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
	
	s1 = input().strip()
	s2 = input().strip()
	
	a, b = s1[0], s1[1]
	a1 = mapping[a]
	b1 = mapping[b]
	diff1 = abs(a1 - b1)
	step1 = min(diff1, 5 - diff1)
	
	c, d = s2[0], s2[1]
	c1 = mapping[c]
	d1 = mapping[d]
	diff2 = abs(c1 - d1)
	step2 = min(diff2, 5 - diff2)
	
	if step1 == step2:
		print("Yes")
	else:
		print("No")

if __name__ == "__main__":
	main()