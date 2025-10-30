import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print("No")
		return
	n = int(data[0])
	S_val = int(data[1])
	A = list(map(int, data[2:2+n]))
	T = sum(A)
	
	k_max = S_val // T
	ceil_val = (S_val + T - 1) // T
	k_min = max(0, ceil_val - 2)
	
	if k_min > k_max:
		print("No")
		return
		
	B = A + A
	
	found = False
	for k in range(k_min, k_max + 1):
		R = S_val - k * T
		if R < 0:
			continue
		if R == 0:
			continue
			
		seen = set()
		seen.add(0)
		current_sum = 0
		for num in B:
			current_sum += num
			if (current_sum - R) in seen:
				found = True
				break
			seen.add(current_sum)
		if found:
			break
			
	print("Yes" if found else "No")

if __name__ == '__main__':
	main()