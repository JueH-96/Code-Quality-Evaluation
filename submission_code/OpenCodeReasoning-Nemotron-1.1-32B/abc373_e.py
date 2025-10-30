import sys
import bisect

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	M = int(data[1])
	K = int(data[2])
	A = list(map(int, data[3:3+n]))
	
	total_votes = sum(A)
	R = K - total_votes
	if R < 0:
		R = 0
		
	B = sorted(A)
	P = [0] * (n+1)
	for i in range(1, n+1):
		P[i] = P[i-1] + B[i-1]
	
	ans = []
	for i in range(n):
		a_i = A[i]
		low, high = 0, R
		res = -1
		while low <= high:
			mid = (low + high) // 2
			V = a_i + mid
			idx1 = bisect.bisect_right(B, V)
			c0 = n - idx1
			if c0 >= M:
				low = mid + 1
				continue
				
			k0 = M - c0
			type2_count = idx1 - 1
			if type2_count < k0:
				res = mid
				high = mid - 1
				continue
				
			S = P[idx1] - P[idx1 - k0]
			if a_i < B[idx1 - k0]:
				sum_excluding = S
			else:
				if idx1 - k0 - 1 < 0:
					sum_excluding = S - a_i
				else:
					sum_excluding = S - a_i + B[idx1 - k0 - 1]
					
			D = R - ( (M - c0) * (a_i + 1) - sum_excluding )
			if D < 0:
				res = mid
				high = mid - 1
				continue
				
			if mid * (M - c0 + 1) > D:
				res = mid
				high = mid - 1
			else:
				low = mid + 1
				
		ans.append(res)
		
	print(" ".join(map(str, ans)))
	
if __name__ == "__main__":
	main()