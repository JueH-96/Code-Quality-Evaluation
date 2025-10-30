base = "wbwbwwbwbwbw"
T = base * 20

def main():
	import sys
	data = sys.stdin.read().split()
	W = int(data[0])
	B = int(data[1])
	L = W + B
	n = len(T)
	
	if L > n:
		print("No")
		return
		
	count_w = 0
	count_b = 0
	for i in range(L):
		if T[i] == 'w':
			count_w += 1
		else:
			count_b += 1
			
	if count_w == W and count_b == B:
		print("Yes")
		return
		
	for start in range(1, n - L + 1):
		if T[start-1] == 'w':
			count_w -= 1
		else:
			count_b -= 1
			
		if T[start+L-1] == 'w':
			count_w += 1
		else:
			count_b += 1
			
		if count_w == W and count_b == B:
			print("Yes")
			return
			
	print("No")

if __name__ == "__main__":
	main()