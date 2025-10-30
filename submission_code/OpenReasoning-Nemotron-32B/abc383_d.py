import math
import bisect

def main():
	N = int(input().strip())
	if N < 36:
		print(0)
		return

	M = int(math.sqrt(N))
	
	if M < 2:
		primes = []
	else:
		sieve = [True] * (M + 1)
		sieve[0] = False
		sieve[1] = False
		sqrtM = int(math.sqrt(M))
		for i in range(2, sqrtM + 1):
			if sieve[i]:
				for j in range(i * i, M + 1, i):
					sieve[j] = False
		primes = [i for i in range(2, M + 1) if sieve[i]]
	
	count1 = 0
	for p in primes:
		p8 = p ** 8
		if p8 > N:
			break
		count1 += 1
		
	count2 = 0
	for i in range(len(primes)):
		p = primes[i]
		if p * p > M:
			break
		q_low = p + 1
		q_high = M // p
		if q_low > q_high:
			continue
		left_index = bisect.bisect_left(primes, q_low)
		if left_index >= len(primes):
			continue
		right_index = bisect.bisect_right(primes, q_high)
		count_here = right_index - left_index
		if count_here > 0:
			count2 += count_here
			
	total = count1 + count2
	print(total)

if __name__ == "__main__":
	main()