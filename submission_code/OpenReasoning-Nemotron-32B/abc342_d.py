import sys
from collections import defaultdict

def main():
	MAX = 200000
	spf = list(range(MAX + 1))
	for i in range(2, int(MAX**0.5) + 1):
		if spf[i] == i:
			for j in range(i * i, MAX + 1, i):
				if spf[j] == j:
					spf[j] = i

	sqf = [0] * (MAX + 1)
	for i in range(1, MAX + 1):
		temp = i
		s = 1
		current = temp
		while current > 1:
			p = spf[current]
			cnt = 0
			while current % p == 0:
				cnt += 1
				current //= p
			if cnt % 2 == 1:
				s *= p
		sqf[i] = s

	data = sys.stdin.read().split()
	n = int(data[0])
	arr = list(map(int, data[1:1 + n]))
	
	zero_count = 0
	freq = defaultdict(int)
	
	for a in arr:
		if a == 0:
			zero_count += 1
		else:
			s_val = sqf[a]
			freq[s_val] += 1
			
	total_pairs = 0
	for count in freq.values():
		total_pairs += count * (count - 1) // 2
		
	total_pairs += zero_count * (zero_count - 1) // 2
	total_pairs += zero_count * (n - zero_count)
	
	print(total_pairs)

if __name__ == "__main__":
	main()