import sys
import bisect

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	
	first_line = data[0].split()
	n = int(first_line[0])
	T = first_line[1].strip()
	strings = []
	for i in range(1, 1 + n):
		strings.append(data[i].strip())
	
	T_rev = T[::-1]
	L_arr = []
	R_arr = []
	
	for s in strings:
		ptr = 0
		for char in s:
			if ptr < len(T) and char == T[ptr]:
				ptr += 1
		L_arr.append(ptr)
		
		ptr2 = 0
		for i in range(len(s) - 1, -1, -1):
			if ptr2 < len(T_rev) and s[i] == T_rev[ptr2]:
				ptr2 += 1
		R_i = len(T) - ptr2
		R_arr.append(R_i)
	
	R_arr.sort()
	total = 0
	for L_i in L_arr:
		idx = bisect.bisect_right(R_arr, L_i)
		total += idx
		
	print(total)

if __name__ == "__main__":
	main()