import sys
from collections import defaultdict

def main():
	S = sys.stdin.readline().strip()
	T = sys.stdin.readline().strip()
	
	U = set("atcoder")
	w_S = S.count('@')
	w_T = T.count('@')
	
	freq_S = defaultdict(int)
	for char in S:
		if char != '@':
			freq_S[char] += 1
			
	freq_T = defaultdict(int)
	for char in T:
		if char != '@':
			freq_T[char] += 1
			
	for char in "abcdefghijklmnopqrstuvwxyz":
		if char not in U:
			if freq_S.get(char, 0) != freq_T.get(char, 0):
				print("No")
				return
				
	total_d = 0
	N = 0
	for char in U:
		count_S = freq_S.get(char, 0)
		count_T = freq_T.get(char, 0)
		d_val = count_T - count_S
		total_d += d_val
		if d_val < 0:
			N += -d_val
			
	if total_d != w_S - w_T:
		print("No")
		return
		
	if N <= w_T:
		print("Yes")
	else:
		print("No")

if __name__ == "__main__":
	main()