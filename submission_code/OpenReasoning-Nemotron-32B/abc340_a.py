def main():
	A, B, D = map(int, input().split())
	n = (B - A) // D + 1
	terms = [str(A + i * D) for i in range(n)]
	print(" ".join(terms))

if __name__ == "__main__":
	main()