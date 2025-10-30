def main():
	mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
	s = input().strip()
	t = input().strip()
	
	a1, a2 = s[0], s[1]
	b1, b2 = t[0], t[1]
	
	diff1 = abs(mapping[a1] - mapping[a2])
	step1 = min(diff1, 5 - diff1)
	
	diff2 = abs(mapping[b1] - mapping[b2])
	step2 = min(diff2, 5 - diff2)
	
	if step1 == step2:
		print("Yes")
	else:
		print("No")

if __name__ == '__main__':
	main()