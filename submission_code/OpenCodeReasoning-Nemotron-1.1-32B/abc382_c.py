import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	A = list(map(int, data[2:2+n]))
	B = list(map(int, data[2+n:2+n+m]))
	
	MAX_VAL = 200000
	min_arr = [10**9] * (MAX_VAL + 1)

	for i in range(n):
		a = A[i]
		if i + 1 < min_arr[a]:
			min_arr[a] = i + 1

	min_index = [10**9] * (MAX_VAL + 1)
	min_index[0] = 10**9
	for x in range(1, MAX_VAL + 1):
		min_index[x] = min(min_index[x - 1], min_arr[x])
		
	for b in B:
		if min_index[b] == 10**9:
			print(-1)
		else:
			print(min_index[b])

if __name__ == '__main__':
	main()