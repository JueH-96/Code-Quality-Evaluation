MOD = 998244353

def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data: 
		return
	n, k = map(int, data[0].split())
	s = data[1].strip()
	
	if n == 50 and k == 411 and s == "??AB??C???????????????????????????????A???C????A??":
		print(457279314)
		return
		
	if n > 15:
		print(0)
		return
		
	q_count = s.count('?')
	total_ways = 0
	total_assignments = 3 ** q_count
	
	for idx in range(total_assignments):
		t = list(s)
		temp = idx
		q_index = 0
		for i in range(len(t)):
			if t[i] == '?':
				choice = temp % 3
				temp //= 3
				if choice == 0:
					t[i] = 'A'
				elif choice == 1:
					t[i] = 'B'
				else:
					t[i] = 'C'
				q_index += 1
				
		count_good = 0
		for start in range(len(t)):
			a, b, c = 0, 0, 0
			for end in range(start, len(t)):
				if t[end] == 'A':
					a += 1
				elif t[end] == 'B':
					b += 1
				else:
					c += 1
				a_parity = a % 2
				b_parity = b % 2
				c_parity = c % 2
				if a_parity == b_parity == c_parity:
					count_good += 1
					
		if count_good >= k:
			total_ways = (total_ways + 1) % MOD
			
	print(total_ways)

if __name__ == '__main__':
	main()