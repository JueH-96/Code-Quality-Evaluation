import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		print("No")
		return
	
	K = int(data[0].strip())
	S = data[1].strip()
	T = data[2].strip()
	
	n = len(S)
	m = len(T)
	
	if abs(n - m) > K:
		print("No")
		return
		
	last_L = max(0, 0 - K)
	last_R = min(m, 0 + K)
	prev = [j for j in range(last_L, last_R + 1)]
	
	for i in range(1, n + 1):
		L_i = max(0, i - K)
		R_i = min(m, i + K)
		curr = [10**9] * (R_i - L_i + 1)
		
		for j in range(L_i, R_i + 1):
			if last_L <= j <= last_R:
				up_val = prev[j - last_L] + 1
			else:
				up_val = 10**9
				
			if j - 1 >= L_i:
				left_val = curr[j - L_i - 1] + 1
			else:
				left_val = 10**9
				
			if j - 1 >= last_L and j - 1 <= last_R:
				cost = 0 if S[i - 1] == T[j - 1] else 1
				diag_val = prev[j - 1 - last_L] + cost
			else:
				diag_val = 10**9
				
			curr[j - L_i] = min(up_val, left_val, diag_val)
			
		prev = curr
		last_L = L_i
		last_R = R_i
		
	if m < last_L or m > last_R:
		ans = 10**9
	else:
		ans = prev[m - last_L]
		
	print("Yes" if ans <= K else "No")

if __name__ == '__main__':
	main()