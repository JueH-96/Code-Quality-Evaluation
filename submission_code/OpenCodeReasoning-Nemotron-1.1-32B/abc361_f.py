import math

def main():
	N = int(input().strip())
	
	if N == 1:
		print(1)
		return
		
	max_k = 0
	temp = 1
	while temp <= N:
		max_k += 1
		temp *= 2
	max_k -= 1
	
	if max_k < 2:
		print(1)
		return
		
	mu = [1] * (max_k + 1)
	is_prime = [True] * (max_k + 1)
	primes = []
	for i in range(2, max_k + 1):
		if is_prime[i]:
			primes.append(i)
			mu[i] = -1
		for p in primes:
			if i * p > max_k:
				break
			is_prime[i * p] = False
			if i % p == 0:
				mu[i * p] = 0
				break
			else:
				mu[i * p] = -mu[i]
				
	total = 0
	for k in range(2, max_k + 1):
		if k == 2:
			count_k = math.isqrt(N)
		else:
			low, high = 1, N
			while low <= high:
				mid = (low + high) // 2
				temp_val = 1
				overflow = False
				for _ in range(k):
					temp_val *= mid
					if temp_val > N:
						overflow = True
						break
				if overflow:
					high = mid - 1
				else:
					low = mid + 1
			count_k = high
		total += mu[k] * (count_k - 1)
		
	answer = 1 - total
	print(answer)

if __name__ == '__main__':
	main()