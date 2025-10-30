import math

def main():
	D = int(input().strip())
	min_diff = D
	x_max = math.isqrt(D)
	
	for x in range(0, x_max + 1):
		target = D - x * x
		y0 = math.isqrt(target)
		
		total1 = x * x + y0 * y0
		if total1 == D:
			min_diff = 0
			break
			
		total2 = x * x + (y0 + 1) * (y0 + 1)
		if total2 == D:
			min_diff = 0
			break
			
		diff1 = abs(total1 - D)
		diff2 = abs(total2 - D)
		current_min = min(diff1, diff2)
		if current_min < min_diff:
			min_diff = current_min
			
	print(min_diff)

if __name__ == '__main__':
	main()