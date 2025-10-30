def main():
	k, g, m = map(int, input().split())
	glass = 0
	mug = 0
	for _ in range(k):
		if glass == g:
			glass = 0
		elif mug == 0:
			mug = m
		else:
			transfer = min(g - glass, mug)
			glass += transfer
			mug -= transfer
	print(f"{glass} {mug}")

if __name__ == "__main__":
	main()