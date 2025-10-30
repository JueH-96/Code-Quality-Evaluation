import sys

MOD = 998244353
MAX_A = 100000

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	phi_arr = list(range(MAX_A + 1))
	for i in range(2, MAX_A + 1):
		if phi_arr[i] == i:
			for j in range(i, MAX_A + 1, i):
				phi_arr[j] -= phi_arr[j] // i
				
	divisors_list = [[] for _ in range(MAX_A + 1)]
	for i in range(1, MAX_A + 1):
		for j in range(i, MAX_A + 1, i):
			divisors_list[j].append(i)
				
	F_prev = 0
	total_count = 0
	F1 = [0] * (MAX_A + 1)
	ans = []
	
	for i in range(n):
		a = A[i]
		if i == 0:
			count_m = 1
			total_count = 1
			for d in divisors_list[a]:
				F1[d] = (F1[d] + 1) % MOD
			ans.append(0)
			F_prev = 0
		else:
			term = 0
			for d in divisors_list[a]:
				term = (term + phi_arr[d] * F1[d]) % MOD
			count_m = (1 + total_count) % MOD
			F_current = (2 * F_prev + term) % MOD
			total_count = (total_count + count_m) % MOD
			for d in divisors_list[a]:
				F1[d] = (F1[d] + count_m) % MOD
			ans.append(F_current)
			F_prev = F_current
			
	for res in ans:
		print(res)

if __name__ == "__main__":
	main()