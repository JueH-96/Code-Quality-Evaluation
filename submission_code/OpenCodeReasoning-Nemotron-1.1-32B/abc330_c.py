import math

def main():
	D = int(input().strip())
	x_max = math.isqrt(D)
	ans = D

	for x in range(0, x_max + 1):
		rem = D - x * x
		y1 = math.isqrt(rem)
		cand1 = rem - y1 * y1
		if cand1 == 0:
			ans = 0
			break
		cand2 = (y1 + 1) * (y1 + 1) - rem
		current_min = min(cand1, cand2)
		if current_min < ans:
			ans = current_min
	else:
		candidate = (x_max + 1) ** 2 - D
		if candidate < ans:
			ans = candidate

	print(ans)

if __name__ == '__main__':
	main()