import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	S = list(map(int, data[1:1+n]))
	
	if n < 3:
		print(0)
		return
		
	T = sorted(S)
	min_val = T[0]
	max_val = T[-1]
	total_range = max_val - min_val + 1
	
	if total_range == n:
		m = (n - 1) // 2
		count = m * (n - (m + 1))
		print(count)
	else:
		count = 0
		for j in range(1, n-1):
			i = j - 1
			k = j + 1
			while i >= 0 and k < n:
				total = T[i] + T[k]
				if total == 2 * T[j]:
					count += 1
					i -= 1
					k += 1
				elif total < 2 * T[j]:
					k += 1
				else:
					i -= 1
		print(count)

if __name__ == "__main__":
	main()