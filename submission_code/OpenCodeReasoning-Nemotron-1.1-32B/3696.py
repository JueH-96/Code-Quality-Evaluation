class Solution:
	def countSubstrings(self, s: str) -> int:
		n = len(s)
		m_map = {1:1, 2:1, 3:3, 4:2, 5:1, 6:3, 7:7, 8:4, 9:9}
		mods_coprime = [1, 3, 7, 9]
		mods_non_coprime = [2, 4]
		
		inv10_dict_coprime = {}
		for m in mods_coprime:
			inv10_dict_coprime[m] = pow(10, -1, m)
		
		P_coprime = {m: 0 for m in mods_coprime}
		freq_coprime = {m: {0: 1} for m in mods_coprime}
		current_power_coprime = {m: 1 for m in mods_coprime}
		
		P2 = 0
		P4_prev = 0
		P4_prev_prev = 0
		
		total = 0
		
		for i in range(n):
			d = int(s[i])
			if d != 0:
				m_val = m_map[d]
				if m_val in mods_coprime:
					if i == 0:
						Q_prev = 0
					else:
						Q_prev = (P_coprime[m_val] * (current_power_coprime[m_val] * 10 % m_val)) % m_val
					count_i = freq_coprime[m_val].get(Q_prev, 0)
					total += count_i
				else:
					if m_val == 2:
						count_i = 1 + (i if (P2 % 2 == 0) else 0)
						total += count_i
					else:
						count_i = 1
						if i - 1 >= 0:
							if (2 * P4_prev_prev) % 4 == P4_prev % 4:
								count_i += 1
						if P4_prev % 4 == 0:
							count_i += (i - 1)
						total += count_i
			
			for m in mods_coprime:
				P_coprime[m] = (P_coprime[m] * 10 + int(s[i])) % m
				Q_i = (P_coprime[m] * current_power_coprime[m]) % m
				if Q_i in freq_coprime[m]:
					freq_coprime[m][Q_i] += 1
				else:
					freq_coprime[m][Q_i] = 1
				current_power_coprime[m] = (current_power_coprime[m] * inv10_dict_coprime[m]) % m
			
			P2 = (P2 * 10 + int(s[i])) % 2
			new_P4 = (P4_prev * 10 + int(s[i])) % 4
			P4_prev_prev = P4_prev
			P4_prev = new_P4
		
		return total