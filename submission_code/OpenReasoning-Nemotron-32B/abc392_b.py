def main():
	n, m = map(int, input().split())
	a_list = list(map(int, input().split()))
	
	set_a = set(a_list)
	missing = []
	for i in range(1, n + 1):
		if i not in set_a:
			missing.append(i)
			
	c = len(missing)
	if c == 0:
		print(0)
	else:
		print(c)
		print(" ".join(map(str, missing)))

if __name__ == "__main__":
	main()