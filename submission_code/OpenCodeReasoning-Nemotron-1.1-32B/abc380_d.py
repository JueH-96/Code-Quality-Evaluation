def main():
	S = input().strip()
	Q = int(input().strip())
	K_list = list(map(int, input().split()))
	
	L0 = len(S)
	if Q == 0:
		print("")
		return
		
	max_k = max(K_list)
	
	n0 = 0
	current_length = L0
	while current_length < max_k:
		n0 += 1
		current_length *= 2
		
	res = []
	for k in K_list:
		cur = k
		swaps = 0
		for i in range(n0, 0, -1):
			half = L0 * (1 << (i-1))
			if cur > half:
				cur -= half
				swaps += 1
		char_at_pos = S[cur-1]
		if swaps % 2 == 1:
			if char_at_pos.islower():
				result_char = char_at_pos.upper()
			else:
				result_char = char_at_pos.lower()
		else:
			result_char = char_at_pos
		res.append(result_char)
		
	print(" ".join(res))

if __name__ == "__main__":
	main()