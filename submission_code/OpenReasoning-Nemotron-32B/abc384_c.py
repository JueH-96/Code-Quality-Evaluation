def main():
	scores = list(map(int, input().split()))
	participants = []
	for mask in range(1, 32):
		total = 0
		name = ""
		for j in range(4, -1, -1):
			if mask & (1 << j):
				idx = 4 - j
				total += scores[idx]
				name += "ABCDE"[idx]
		participants.append((total, name))
	
	participants.sort(key=lambda x: (-x[0], x[1]))
	
	for _, name in participants:
		print(name)

if __name__ == '__main__':
	main()