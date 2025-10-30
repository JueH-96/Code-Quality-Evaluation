import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	k = int(data[1])
	A = list(map(int, data[2:2+n]))
	
	MAX_VAL = 1000000
	freq = [0] * (MAX_VAL + 1)
	for a in A:
		if a <= MAX_VAL:
			freq[a] += 1

	count_multiples = [0] * (MAX_VAL + 1)
	for d in range(1, MAX_VAL + 1):
		total = 0
		j = d
		while j <= MAX_VAL:
			total += freq[j]
			j += d
		count_multiples[d] = total

	ans_value = [0] * (MAX_VAL + 1)
	for d in range(MAX_VAL, 0, -1):
		if count_multiples[d] >= k:
			j = d
			while j <= MAX_VAL:
				if ans_value[j] == 0:
					ans_value[j] = d
				j += d

	for a in A:
		print(ans_value[a])

if __name__ == "__main__":
	main()