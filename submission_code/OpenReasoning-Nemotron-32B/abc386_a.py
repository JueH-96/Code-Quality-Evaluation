from collections import Counter

def main():
	data = input().split()
	nums = list(map(int, data))
	cnt = Counter(nums)
	freqs = sorted(cnt.values())
	if freqs == [1, 3] or freqs == [2, 2]:
		print("Yes")
	else:
		print("No")

if __name__ == "__main__":
	main()