def main():
	n, m = map(int, input().split())
	a = list(map(int, input().split()))
	
	s = set(a)
	missing = []
	for i in range(1, n + 1):
		if i not in s:
			missing.append(i)
			
	if not missing:
		print(0)
	else:
		print(len(missing))
		print(" ".join(map(str, missing)))

if __name__ == "__main__":
	main()