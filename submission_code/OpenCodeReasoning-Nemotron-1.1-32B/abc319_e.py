import math
import sys

def lcm(a, b):
	return a * b // math.gcd(a, b)

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	it = iter(data)
	n = int(next(it))
	X = int(next(it))
	Y = int(next(it))
	P = []
	T = []
	for _ in range(n - 1):
		p = int(next(it))
		t_val = int(next(it))
		P.append(p)
		T.append(t_val)
	
	distinct_P = set(P)
	L = 1
	for p in distinct_P:
		L = lcm(L, p)
	
	current = [r + X for r in range(L)]
	
	for i in range(n - 1):
		p_i = P[i]
		t_i = T[i]
		for r in range(L):
			t_val = current[r]
			rem = t_val % p_i
			if rem:
				current[r] = t_val + p_i - rem + t_i
			else:
				current[r] = t_val + t_i
	
	for r in range(L):
		current[r] += Y
	
	F = current
	Q = int(next(it))
	out_lines = []
	for _ in range(Q):
		q = int(next(it))
		base = q // L
		r = q % L
		ans = base * L + F[r]
		out_lines.append(str(ans))
	
	print("
".join(out_lines))

if __name__ == "__main__":
	main()