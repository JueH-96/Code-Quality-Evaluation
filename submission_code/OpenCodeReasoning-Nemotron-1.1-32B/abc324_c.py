import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	
	first_line = data[0].split()
	if not first_line:
		print(0)
		return
		
	n = int(first_line[0])
	t_prime = first_line[1].strip()
	
	strings = []
	for i in range(1, 1 + n):
		strings.append(data[i].strip())
	
	ans = []
	
	for idx, s in enumerate(strings):
		len_s = len(s)
		len_t = len(t_prime)
		
		if len_s == len_t:
			if s == t_prime:
				ans.append(idx + 1)
			else:
				diff_count = 0
				for j in range(len_s):
					if s[j] != t_prime[j]:
						diff_count += 1
						if diff_count > 1:
							break
				if diff_count == 1:
					ans.append(idx + 1)
					
		elif len_s == len_t - 1:
			i_ptr, j_ptr = 0, 0
			skipped = 0
			n_t = len_t
			m_s = len_s
			while i_ptr < n_t and j_ptr < m_s:
				if t_prime[i_ptr] == s[j_ptr]:
					i_ptr += 1
					j_ptr += 1
				else:
					if skipped == 0:
						skipped = 1
						i_ptr += 1
					else:
						break
			if j_ptr == m_s and (i_ptr == n_t or (skipped == 0 and i_ptr == n_t - 1)):
				ans.append(idx + 1)
				
		elif len_s == len_t + 1:
			i_ptr, j_ptr = 0, 0
			skipped = 0
			n_t = len_t
			m_s = len_s
			while i_ptr < m_s and j_ptr < n_t:
				if s[i_ptr] == t_prime[j_ptr]:
					i_ptr += 1
					j_ptr += 1
				else:
					if skipped == 0:
						skipped = 1
						i_ptr += 1
					else:
						break
			if j_ptr == n_t and (i_ptr == m_s or (skipped == 0 and i_ptr == m_s - 1)):
				ans.append(idx + 1)
				
	print(len(ans))
	if ans:
		print(" ".join(map(str, ans)))

if __name__ == "__main__":
	main()