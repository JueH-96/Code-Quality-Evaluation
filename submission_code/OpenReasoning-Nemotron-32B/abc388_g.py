import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	q = int(data[1+n])
	queries = []
	index = 1+n+1
	for i in range(q):
		L = int(data[index])
		R = int(data[index+1])
		index += 2
		queries.append((L, R))
	
	out_lines = []
	for (L, R) in queries:
		L0 = L - 1
		R0 = R - 1
		m = R0 - L0 + 1
		small_count = m // 2
		large_start = L0 + small_count
		count = 0
		i_ptr = L0
		for j in range(large_start, R0 + 1):
			if i_ptr >= L0 + small_count:
				break
			if A[i_ptr] <= A[j] / 2.0:
				count += 1
				i_ptr += 1
		out_lines.append(str(count))
	
	sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
	main()