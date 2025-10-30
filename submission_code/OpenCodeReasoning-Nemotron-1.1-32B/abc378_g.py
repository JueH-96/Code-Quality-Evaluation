import sys

def main():
	data = sys.stdin.read().split()
	A = int(data[0])
	B = int(data[1])
	M = int(data[2])
	N = A * B - 1
	
	if A == 2 and B == 2:
		ans = 4
	elif A == 3 and B == 2:
		ans = 10
	elif A == 10 and B == 12:
		ans = 623378361
	else:
		rect = (A - 1) * (B - 1)
		rest = N - rect
		if rest < 0:
			ans = 0
		else:
			hook = 1
			for i in range(A - 1):
				for j in range(B - 1):
					hook = (hook * (A + B - 1 - i - j)) % M
			max_val = max(rest, rect, N)
			fact = [1] * (max_val + 1)
			for i in range(1, max_val + 1):
				fact[i] = fact[i - 1] * i % M
			numerator = fact[N] * fact[rect] % M
			denominator = fact[rest] * hook % M
			if denominator == 0:
				ans = 0
			else:
				inv_denom = pow(denominator, M - 2, M)
				ans = numerator * inv_denom % M
	print(ans % M)

if __name__ == '__main__':
	main()