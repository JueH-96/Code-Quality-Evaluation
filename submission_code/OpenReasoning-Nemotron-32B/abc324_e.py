import sys
import bisect

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	
	parts = data[0].split()
	if not parts:
		print(0)
		return
		
	N = int(parts[0])
	T_str = parts[1].strip()
	
	S_list = []
	for i in range(1, 1+N):
		S_list.append(data[i].strip())
	
	L = len(T_str)
	
	A = []
	for s in S_list:
		k = 0
		for char in s:
			if k < L and char == T_str[k]:
				k += 1
		A.append(k)
	
	B = []
	for s in S_list:
		j = L - 1
		for idx in range(len(s)-1, -1, -1):
			if j >= 0 and s[idx] == T_str[j]:
				j -= 1
		B.append(j+1)
	
	A.sort()
	total = 0
	for b_val in B:
		idx = bisect.bisect_left(A, b_val)
		total += (len(A) - idx)
		
	print(total)

if __name__ == "__main__":
	main()