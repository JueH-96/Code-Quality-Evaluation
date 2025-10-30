def main():
	n = int(input().strip())
	s = input().strip()
	
	total_t = s.count('T')
	total_a = s.count('A')
	
	if total_t > total_a:
		print('T')
	elif total_a > total_t:
		print('A')
	else:
		indexT = None
		indexA = None
		countT = 0
		countA = 0
		for i, char in enumerate(s):
			if char == 'T':
				countT += 1
				if countT == total_t and indexT is None:
					indexT = i
			else:
				countA += 1
				if countA == total_a and indexA is None:
					indexA = i
			if indexT is not None and indexA is not None:
				break
		if indexT < indexA:
			print('T')
		else:
			print('A')

if __name__ == "__main__":
	main()