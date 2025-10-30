import sys
from collections import defaultdict

def main():
	data = sys.stdin.read().splitlines()
	s = data[0].strip()
	t = data[1].strip()
	V = set('atcoder')
	a = s.count('@')
	b = t.count('@')
	
	count_s = defaultdict(int)
	count_t = defaultdict(int)
	
	for char in s:
		if char != '@':
			count_s[char] += 1
			
	for char in t:
		if char != '@':
			count_t[char] += 1
			
	keys = set(count_s.keys()) | set(count_t.keys())
	
	for char in keys:
		if char not in V:
			if count_s.get(char, 0) != count_t.get(char, 0):
				print("No")
				return
				
	D_total = 0
	L = 0
	for char in V:
		cnt_s = count_s.get(char, 0)
		cnt_t = count_t.get(char, 0)
		D_total += (cnt_s - cnt_t)
		L += max(0, cnt_t - cnt_s)
		
	if a + D_total == b and L <= a:
		print("Yes")
	else:
		print("No")

if __name__ == '__main__':
	main()