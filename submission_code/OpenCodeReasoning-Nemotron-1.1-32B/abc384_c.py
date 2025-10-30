def main():
	a, b, c, d, e = map(int, input().split())
	letters = ['A', 'B', 'C', 'D', 'E']
	scores = [a, b, c, d, e]
	
	participants = []
	for bitmask in range(1, 32):
		name = ""
		total_score = 0
		for j in range(5):
			if bitmask & (1 << j):
				name += letters[j]
				total_score += scores[j]
		participants.append((name, total_score))
	
	participants.sort(key=lambda x: (-x[1], x[0]))
	
	for name, _ in participants:
		print(name)

if __name__ == "__main__":
	main()