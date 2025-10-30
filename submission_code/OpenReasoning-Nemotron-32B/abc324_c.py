import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	first_line = data[0].split()
	if len(first_line) < 2:
		print(0)
		return
	try:
		N_val = int(first_line[0])
		T_prime = first_line[1]
	except:
		print(0)
		return
		
	S_list = []
	for i in range(1, 1 + N_val):
		if i < len(data):
			S_list.append(data[i].strip())
		else:
			S_list.append("")
	
	L = len(T_prime)
	results = []
	
	for idx, s in enumerate(S_list, start=1):
		n = len(s)
		if n == L:
			if s == T_prime:
				results.append(idx)
			else:
				diff_count = 0
				for i_char in range(L):
					if s[i_char] != T_prime[i_char]:
						diff_count += 1
						if diff_count > 1:
							break
				if diff_count == 1:
					results.append(idx)
		elif n == L + 1:
			i_ptr, j_ptr = 0, 0
			skipped = 0
			while j_ptr < L and i_ptr < n:
				if s[i_ptr] == T_prime[j_ptr]:
					i_ptr += 1
					j_ptr += 1
				else:
					if skipped == 0:
						skipped = 1
						i_ptr += 1
					else:
						break
			if j_ptr == L:
				if (skipped == 1 and i_ptr == n) or (skipped == 0 and i_ptr == n - 1):
					results.append(idx)
		elif n == L - 1:
			i_ptr, j_ptr = 0, 0
			skipped = 0
			while i_ptr < n and j_ptr < L:
				if s[i_ptr] == T_prime[j_ptr]:
					i_ptr += 1
					j_ptr += 1
				else:
					if skipped == 0:
						skipped = 1
						j_ptr += 1
					else:
						break
			if i_ptr == n:
				if (skipped == 1 and j_ptr == L) or (skipped == 0 and j_ptr == L - 1):
					results.append(idx)
					
	k = len(results)
	print(k)
	if k > 0:
		print(" ".join(map(str, results)))

if __name__ == "__main__":
	main()