import math

def integer_kth_root(N, k):
	if k == 1:
		return N
	if k == 2:
		return math.isqrt(N)
	low, high = 1, 1
	while True:
		if high > 10**7:
			break
		try:
			power = pow(high, k)
		except OverflowError:
			break
		if power < N:
			low = high
			high *= 2
		elif power == N:
			return high
		else:
			break
	if high > 10**7:
		high = 10**7
	while low <= high:
		mid = (low + high) // 2
		try:
			mid_power = pow(mid, k)
		except OverflowError:
			high = mid - 1
			continue
		if mid_power == N:
			return mid
		elif mid_power < N:
			low = mid + 1
		else:
			high = mid - 1
	return high

def main():
	max_k = 60
	mobius = [1] * (max_k + 1)
	is_prime = [True] * (max_k + 1)
	primes = []
	for i in range(2, max_k + 1):
		if is_prime[i]:
			primes.append(i)
			mobius[i] = -1
		for p in primes:
			if i * p > max_k:
				break
			is_prime[i * p] = False
			if i % p == 0:
				mobius[i * p] = 0
				break
			else:
				mobius[i * p] = -mobius[i]
	
	N = int(input().strip())
	total = 0
	for k in range(2, 61):
		root = integer_kth_root(N, k)
		if root <= 1:
			term = 0
		else:
			term = mobius[k] * (root - 1)
		total += term
	answer = 1 - total
	print(answer)

if __name__ == '__main__':
	main()