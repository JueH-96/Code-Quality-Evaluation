import sys

def main():
	data = sys.stdin.read().splitlines()
	t = int(data[0])
	index = 1
	out_lines = []
	for _ in range(t):
		n = int(data[index])
		index += 1
		A = data[index].strip()
		index += 1
		B = data[index].strip()
		index += 1
		
		total_B = 0
		L_B = None
		R_B = None
		for i, b_char in enumerate(B):
			if b_char == '1':
				total_B += 1
				if L_B is None:
					L_B = i + 1
				R_B = i + 1
		
		total_A = A.count('1')
		
		if total_A < total_B:
			out_lines.append("-1")
			continue
		
		left_needed = 0
		right_needed = 0
		for i, a_char in enumerate(A):
			if a_char == '1':
				pos = i + 1
				if pos < L_B:
					left_needed = max(left_needed, L_B - pos)
				elif pos > R_B:
					right_needed = max(right_needed, pos - R_B)
		
		ans = max(left_needed, right_needed)
		out_lines.append(str(ans))
	
	print("
".join(out_lines))

if __name__ == "__main__":
	main()