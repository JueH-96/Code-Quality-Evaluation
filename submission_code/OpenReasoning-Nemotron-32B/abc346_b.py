def main():
	import sys
	data = sys.stdin.read().split()
	W = int(data[0])
	B = int(data[1])
	total_chars = W + B
	base = "wbwbwwbwbwbw"
	s = base * 20
	
	n = len(s)
	found = False
	for i in range(0, n - total_chars + 1):
		substr = s[i:i+total_chars]
		count_w = substr.count('w')
		count_b = substr.count('b')
		if count_w == W and count_b == B:
			found = True
			break
			
	print("Yes" if found else "No")

if __name__ == "__main__":
	main()