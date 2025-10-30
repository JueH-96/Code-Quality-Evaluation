import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		return
	nq_line = data[0].split()
	N = int(nq_line[0])
	Q = int(nq_line[1])
	S = data[1].strip()
	base_S = [int(char) for char in S]
	
	fenw_flip = [0] * (N + 2)
	
	if N >= 2:
		B_arr = []
		for i in range(N - 1):
			if base_S[i] == base_S[i + 1]:
				B_arr.append(1)
			else:
				B_arr.append(0)
	else:
		B_arr = []
	
	fenw_B = [0] * (N + 1)
	
	def update_fenw(fenw, size, index, delta):
		idx = index + 1
		while idx <= size:
			fenw[idx] += delta
			idx += idx & -idx
			
	def query_fenw(fenw, index):
		if index < 0:
			return 0
		total = 0
		idx = index + 1
		while idx > 0:
			total += fenw[idx]
			idx -= idx & -idx
		return total

	def update_B(i, delta):
		update_fenw(fenw_B, N, i, delta)
		
	def query_B(i):
		return query_fenw(fenw_B, i)
	
	for i in range(len(B_arr)):
		update_B(i, B_arr[i])
	
	out_lines = []
	index = 2
	for _ in range(Q):
		parts = data[index].split()
		index += 1
		if parts[0] == '1':
			L = int(parts[1])
			R = int(parts[2])
			l0 = L - 1
			r0 = R - 1
			if l0 - 1 >= 0 and l0 - 1 < N - 1:
				flips_l0_minus_1 = query_fenw(fenw_flip, l0 - 1) % 2
				flips_l0 = query_fenw(fenw_flip, l0) % 2
				char_l0_minus_1 = base_S[l0 - 1] ^ flips_l0_minus_1
				char_l0 = base_S[l0] ^ flips_l0
				new_char_l0 = 1 - char_l0
				old_val = 1 if char_l0_minus_1 == char_l0 else 0
				new_val = 1 if char_l0_minus_1 == new_char_l0 else 0
				diff = new_val - old_val
				update_B(l0 - 1, diff)
			if r0 < N - 1:
				flips_r0 = query_fenw(fenw_flip, r0) % 2
				flips_r0_plus_1 = query_fenw(fenw_flip, r0 + 1) % 2
				char_r0 = base_S[r0] ^ flips_r0
				char_r0_plus_1 = base_S[r0 + 1] ^ flips_r0_plus_1
				new_char_r0 = 1 - char_r0
				old_val = 1 if char_r0 == char_r0_plus_1 else 0
				new_val = 1 if new_char_r0 == char_r0_plus_1 else 0
				diff = new_val - old_val
				update_B(r0, diff)
			update_fenw(fenw_flip, N + 1, l0, 1)
			if r0 + 1 < N:
				update_fenw(fenw_flip, N + 1, r0 + 1, 1)
		else:
			L = int(parts[1])
			R = int(parts[2])
			l0 = L - 1
			r0 = R - 1
			if l0 == r0:
				out_lines.append("Yes")
			else:
				total_bad = query_B(r0 - 1) - query_B(l0 - 1)
				if total_bad == 0:
					out_lines.append("Yes")
				else:
					out_lines.append("No")
					
	for line in out_lines:
		print(line)

if __name__ == "__main__":
	main()