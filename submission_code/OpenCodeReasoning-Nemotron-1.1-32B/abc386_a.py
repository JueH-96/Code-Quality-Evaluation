from collections import Counter

def main():
	data = input().split()
	A, B, C, D = map(int, data)
	s = set([A, B, C, D])
	ans = "No"
	for x in s:
		hand = [A, B, C, D, x]
		cnt = Counter(hand)
		if len(cnt) == 2:
			counts = sorted(cnt.values())
			if counts == [2, 3]:
				ans = "Yes"
				break
	print(ans)

if __name__ == "__main__":
	main()