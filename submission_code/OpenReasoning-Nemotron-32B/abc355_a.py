def main():
	A, B = map(int, input().split())
	s = {1, 2, 3}
	s.discard(A)
	s.discard(B)
	if len(s) == 1:
		print(s.pop())
	else:
		print(-1)

if __name__ == "__main__":
	main()