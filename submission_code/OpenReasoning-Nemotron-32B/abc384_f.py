import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	pairs = []
	for x in A:
		v = 0
		temp = x
		while temp % 2 == 0:
			v += 1
			temp //= 2
		pairs.append((v, temp))
	
	groups = {}
	for v, a in pairs:
		if v not in groups:
			groups[v] = []
		groups[v].append(a)
	
	keys = sorted(groups.keys())
	entire_S1 = 0
	for i in range(len(keys)):
		k = keys[i]
		group_k = groups[k]
		count_k = len(group_k)
		S_k = sum(group_k)
		for j in range(len(keys)):
			if i == j:
				continue
			l = keys[j]
			group_l = groups[l]
			count_l = len(group_l)
			S_l = sum(group_l)
			if k < l:
				d = l - k
				term = count_l * S_k + (1 << d) * count_k * S_l
			else:
				d = k - l
				term = count_k * S_l + (1 << d) * count_l * S_k
			entire_S1 += term
			
	S1_upper = entire_S1 // 2
	
	S2_upper = 0
	for k, arr in groups.items():
		m = len(arr)
		if m == 0:
			continue
		S_group = sum(arr)
		diagonal = S_group
		off_diagonal = 0
		if m >= 2:
			T = [0] * 26
			T[1] = (m - 1) * S_group
			for v in range(2, 26):
				mod = 1 << v
				count_dict = {}
				total_dict = {}
				for a in arr:
					r = a % mod
					count_dict[r] = count_dict.get(r, 0) + 1
					total_dict[r] = total_dict.get(r, 0) + a
				t_val = 0
				for r, total_val in total_dict.items():
					s = (mod - r) % mod
					if s in count_dict:
						t_val += total_val * count_dict[s]
				T[v] = t_val
			for v in range(1, 26):
				if v < 25:
					exact = T[v] - T[v+1]
				else:
					exact = T[25]
				off_diagonal += exact // (1 << v)
		S2_upper += diagonal + off_diagonal
		
	ans = S1_upper + S2_upper
	print(ans)

if __name__ == "__main__":
	main()