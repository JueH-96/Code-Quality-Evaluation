import sys

def main():
	data = sys.stdin.readline().split()
	if not data:
		return
	N = int(data[0])
	L = int(data[1])
	R = int(data[2])
	total = 0
	l = L
	r = R

	while l <= r:
		if l == 0:
			k0 = N
		else:
			lowbit = l & -l
			k0 = lowbit.bit_length() - 1
		length = r - l + 1
		k_max = length.bit_length() - 1
		k = min(k0, k_max)
		seg_length = 1 << k
		j = l // seg_length
		print(f"? {k} {j}")
		sys.stdout.flush()
		T = int(sys.stdin.readline().strip())
		if T == -1:
			return
		total = (total + T) % 100
		l += seg_length

	print(f"! {total}")
	sys.stdout.flush()

if __name__ == '__main__':
	main()