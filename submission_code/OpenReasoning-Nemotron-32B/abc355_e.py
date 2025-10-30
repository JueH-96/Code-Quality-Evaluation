import sys

def main():
	data = sys.stdin.readline().split()
	if not data:
		return
	N = int(data[0])
	L = int(data[1])
	R = int(data[2])
	total_size = 1 << N

	def decompose(segl, segr, l, r):
		if segl > r or segr < l:
			return 0
		if l <= segl and segr <= r:
			size = segr - segl + 1
			i = size.bit_length() - 1
			j = segl >> i
			print(f"? {i} {j}", flush=True)
			response = sys.stdin.readline().strip()
			if response == '':
				sys.exit(0)
			T = int(response)
			if T == -1:
				sys.exit(0)
			return T
		mid = (segl + segr) // 2
		left_sum = decompose(segl, mid, l, r)
		right_sum = decompose(mid + 1, segr, l, r)
		return (left_sum + right_sum) % 100

	ans = decompose(0, total_size - 1, L, R)
	print(f"! {ans}", flush=True)

if __name__ == "__main__":
	main()