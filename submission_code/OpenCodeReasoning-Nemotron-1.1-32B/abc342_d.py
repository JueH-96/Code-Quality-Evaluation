import math

MAX_VAL = 200000

min_prime = list(range(MAX_VAL + 1))

for i in range(2, int(math.isqrt(MAX_VAL)) + 1):
	if min_prime[i] == i:
		for j in range(i * i, MAX_VAL + 1, i):
			if min_prime[j] == j:
				min_prime[j] = i

f = [0] * (MAX_VAL + 1)
f[1] = 1
for i in range(2, MAX_VAL + 1):
	p = min_prime[i]
	q = i // p
	if q % p == 0:
		f[i] = f[q // p]
	else:
		f[i] = f[q] * p

def main():
	n = int(input().strip())
	arr = list(map(int, input().split()))
	
	count0 = 0
	freq_arr = [0] * (MAX_VAL + 1)
	
	for a in arr:
		if a == 0:
			count0 += 1
		else:
			kernel_val = f[a]
			freq_arr[kernel_val] += 1
			
	total_pairs = count0 * (count0 - 1) // 2 + count0 * (n - count0)
	for count in freq_arr:
		total_pairs += count * (count - 1) // 2
		
	print(total_pairs)

if __name__ == '__main__':
	main()