import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(-1)
		return
	
	M = int(data[0].strip())
	S0 = data[1].strip()
	S1 = data[2].strip()
	S2 = data[3].strip()
	
	residues = [dict() for _ in range(3)]
	reels = [S0, S1, S2]
	
	for i in range(3):
		s = reels[i]
		for idx, char in enumerate(s):
			if char not in residues[i]:
				residues[i][char] = []
			residues[i][char].append(idx)
	
	T_max = 3 * M
	
	for T in range(0, T_max + 1):
		for c in '0123456789':
			if c not in residues[0] or c not in residues[1] or c not in residues[2]:
				continue
				
			sets = []
			for i in range(3):
				a_list = residues[i][c]
				s_set = set()
				for a in a_list:
					t_val = a
					while t_val <= T:
						s_set.add(t_val)
						t_val += M
				sets.append(s_set)
				
			set0, set1, set2 = sets
			found = False
			for t0 in set0:
				for t1 in set1:
					if t1 == t0:
						continue
					for t2 in set2:
						if t2 == t0 or t2 == t1:
							continue
						found = True
						break
					if found:
						break
				if found:
					break
					
			if found:
				print(T)
				return
				
	print(-1)

if __name__ == "__main__":
	main()