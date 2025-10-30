import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print("No")
		return
	try:
		K = int(data[0].strip())
		S = data[1].strip()
		T = data[2].strip()
	except Exception:
		print("No")
		return

	n = len(S)
	m = len(T)
	
	if abs(n - m) > K:
		print("No")
		return
		
	L_prev = 0
	R_prev = min(m, K)
	dp_prev = [0] * (R_prev - L_prev + 1)
	for j in range(L_prev, R_prev + 1):
		dp_prev[j - L_prev] = j
		
	if n == 0:
		if m <= K:
			if dp_prev[m - L_prev] <= K:
				print("Yes")
			else:
				print("No")
		else:
			print("No")
		return

	for i in range(1, n + 1):
		L_curr = max(0, i - K)
		R_curr = min(m, i + K)
		curr = [10**9] * (R_curr - L_curr + 1)
		
		for j in range(L_curr, R_curr + 1):
			idx_curr = j - L_curr
			candidate1 = 10**9
			if L_prev <= j <= R_prev:
				idx_prev = j - L_prev
				candidate1 = dp_prev[idx_prev] + 1
				
			candidate2 = 10**9
			if j - 1 >= L_curr and j - 1 <= R_curr:
				idx_prev_curr = (j - 1) - L_curr
				candidate2 = curr[idx_prev_curr] + 1
				
			candidate3 = 10**9
			if j - 1 >= L_prev and j - 1 <= R_prev:
				idx_prev_j1 = (j - 1) - L_prev
				cost = 0 if S[i - 1] == T[j - 1] else 1
				candidate3 = dp_prev[idx_prev_j1] + cost
				
			curr[idx_curr] = min(candidate1, candidate2, candidate3)
			
		if min(curr) > K:
			print("No")
			return
			
		dp_prev = curr
		L_prev = L_curr
		R_prev = R_curr
		
	if m < L_prev or m > R_prev:
		print("No")
	else:
		idx_last = m - L_prev
		if dp_prev[idx_last] <= K:
			print("Yes")
		else:
			print("No")

if __name__ == '__main__':
	main()