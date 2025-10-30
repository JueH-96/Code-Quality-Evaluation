import sys

def main():
	data = sys.stdin.read().split()
	N = int(data[0])
	K = int(data[1])
	nk = N * K
	counts = [0] + [K] * N
	T = 1
	for i in range(1, nk + 1):
		T = T * i
	factK = 1
	for i in range(1, K + 1):
		factK = factK * i
	T //= factK ** N
	L = nk
	M = (T + 1) // 2
	result = []
	for _ in range(nk):
		base = T // L
		rem = T % L
		for num in range(1, N + 1):
			if counts[num] == 0:
				continue
			count_i = base * counts[num] + (rem * counts[num]) // L
			if M <= count_i:
				result.append(str(num))
				T = count_i
				counts[num] -= 1
				L -= 1
				break
			else:
				M -= count_i
	print(" ".join(result))

if __name__ == '__main__':
	main()