import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print("No")
		return
	n = int(data[0].strip())
	intervals = []
	total_min = 0
	total_max = 0
	for i in range(1, n + 1):
		if i >= len(data):
			break
		line = data[i].split()
		if not line:
			continue
		L = int(line[0])
		R = int(line[1])
		intervals.append((L, R))
		total_min += L
		total_max += R
		
	if total_min <= 0 <= total_max:
		remaining = -total_min
		res = []
		for L, R in intervals:
			add = min(remaining, R - L)
			res.append(L + add)
			remaining -= add
		print("Yes")
		print(" ".join(map(str, res)))
	else:
		print("No")

if __name__ == "__main__":
	main()