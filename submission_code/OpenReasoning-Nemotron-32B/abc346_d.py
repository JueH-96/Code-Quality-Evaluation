import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		return
	N = int(data[0].strip())
	S = data[1].strip()
	C = list(map(int, data[2].split()))
	
	base0_arr = []
	base1_arr = []
	for j in range(N):
		if j % 2 == 0:
			base0_arr.append('0')
			base1_arr.append('1')
		else:
			base0_arr.append('1')
			base1_arr.append('0')
			
	P0 = [0] * (N+1)
	for j in range(1, N+1):
		if S[j-1] != base0_arr[j-1]:
			P0[j] = P0[j-1] + C[j-1]
		else:
			P0[j] = P0[j-1]
			
	P1 = [0] * (N+1)
	for j in range(1, N+1):
		if S[j-1] != base1_arr[j-1]:
			P1[j] = P1[j-1] + C[j-1]
		else:
			P1[j] = P1[j-1]
			
	P0_total = P0[N]
	P1_total = P1[N]
	
	ans = 10**18
	
	for base_pattern in [0, 1]:
		for i in range(0, N-1):
			for x in ['0', '1']:
				if i >= 1:
					if base_pattern == 0:
						if base0_arr[i-1] == x:
							continue
					else:
						if base1_arr[i-1] == x:
							continue
							
				if base_pattern == 0:
					part1 = P0[i]
				else:
					part1 = P1[i]
					
				cost_i = 0
				if S[i] != x:
					cost_i += C[i]
				if S[i+1] != x:
					cost_i += C[i+1]
					
				p = i + 2
				if p < N:
					y0 = '1' if x == '0' else '0'
					if base0_arr[p] == y0:
						part3 = P0_total - P0[p]
					else:
						part3 = P1_total - P1[p]
				else:
					part3 = 0
					
				total_cost = part1 + cost_i + part3
				if total_cost < ans:
					ans = total_cost
					
	print(ans)

if __name__ == '__main__':
	main()