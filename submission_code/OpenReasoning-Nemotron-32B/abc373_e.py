import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	M = int(data[1])
	K = int(data[2])
	A = list(map(int, data[3:3+n]))
	
	total_votes = sum(A)
	rem = K - total_votes
	if rem < 0:
		print(" ".join(["-1"] * n))
		return
		
	b = sorted(A, reverse=True)
	d = [0] * (n+1)
	for i in range(1, n+1):
		d[i] = d[i-1] + b[i-1]
	
	ans = [-1] * n

	def check(i, X):
		T = A[i] + X
		lo, hi = 0, n
		while lo < hi:
			mid = (lo + hi) // 2
			if b[mid] > T:
				lo = mid + 1
			else:
				hi = mid
		count_above = lo
		
		if count_above >= M:
			return False
			
		k0 = M - count_above
		if k0 <= 0:
			return True
			
		total_in_set = n - count_above
		if k0 > total_in_set - 1:
			if n - 1 < M:
				return True
			else:
				return False
				
		start_index = count_above
		end_index = count_above + k0 - 1
		S_total = d[end_index+1] - d[start_index]
		
		if A[i] >= b[end_index]:
			if count_above + k0 < n:
				next_val = b[count_above + k0]
			else:
				next_val = 0
			S_without_i = S_total - A[i] + next_val
		else:
			S_without_i = S_total
			
		requirement_sum = k0 * (T + 1) - S_without_i
		if requirement_sum > rem - X:
			return True
		else:
			return False

	for i in range(n):
		low_bound = max(0, b[M-1] - A[i])
		if low_bound > rem:
			ans[i] = -1
		else:
			left = low_bound
			right = rem
			while left < right:
				mid = (left + right) // 2
				if check(i, mid):
					right = mid
				else:
					left = mid + 1
			if check(i, left):
				ans[i] = left
			else:
				ans[i] = -1
				
	print(" ".join(map(str, ans)))
	
if __name__ == "__main__":
	main()