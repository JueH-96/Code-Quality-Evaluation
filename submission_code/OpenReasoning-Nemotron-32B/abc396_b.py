def main():
	stack = [0] * 100
	q = int(input().strip())
	outputs = []
	for _ in range(q):
		data = input().split()
		if data[0] == '1':
			x = int(data[1])
			stack.append(x)
		else:
			card = stack.pop()
			outputs.append(card)
	
	for card in outputs:
		print(card)

if __name__ == "__main__":
	main()