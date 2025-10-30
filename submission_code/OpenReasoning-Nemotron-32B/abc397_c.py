import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	left_arr = [0] * (n + 1)
	freq_left = [0] * (n + 1)
	distinct = 0
	for i in range(n):
		num = A[i]
		freq_left[num] += 1
		if freq_left[num] == 1:
			distinct += 1
		left_arr[i + 1] = distinct
		
	right_arr = [0] * (n + 1)
	freq_right = [0] * (n + 1)
	distinct = 0
	for i in range(n - 1, -1, -1):
		num = A[i]
		freq_right[num] += 1
		if freq_right[num] == 1:
			distinct += 1
		right_arr[i] = distinct
		
	ans = 0
	for i in range(1, n):
		total = left_arr[i] + right_arr[i]
		if total > ans:
			ans = total
			
	print(ans)

if __name__ == "__main__":
	main()