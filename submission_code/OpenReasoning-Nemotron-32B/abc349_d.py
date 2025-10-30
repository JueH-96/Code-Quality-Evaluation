def main():
	import sys
	data = sys.stdin.read().split()
	L = int(data[0])
	R = int(data[1])
	segments = []
	current = L
	
	while current < R:
		if current == 0:
			if R == 0:
				break
			step = 1 << (R.bit_length() - 1)
		else:
			lowbit = current & -current
			k0 = lowbit.bit_length() - 1
			n = R - current
			if n == 0:
				break
			i_max = n.bit_length() - 1
			i = min(k0, i_max)
			step = 1 << i
		
		segments.append((current, current + step))
		current = current + step
	
	print(len(segments))
	for a, b in segments:
		print(a, b)

if __name__ == "__main__":
	main()