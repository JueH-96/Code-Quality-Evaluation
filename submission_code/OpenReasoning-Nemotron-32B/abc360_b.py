def main():
	data = input().split()
	S = data[0]
	T = data[1]
	n = len(S)
	
	for w in range(1, n):
		chunks = []
		start = 0
		while start < n:
			end = start + w
			chunks.append(S[start:end])
			start = end
		
		for c in range(1, w + 1):
			built = []
			for chunk in chunks:
				if len(chunk) >= c:
					built.append(chunk[c - 1])
					if len(built) > len(T):
						break
			if len(built) != len(T):
				continue
			if ''.join(built) == T:
				print("Yes")
				return
	print("No")

if __name__ == "__main__":
	main()