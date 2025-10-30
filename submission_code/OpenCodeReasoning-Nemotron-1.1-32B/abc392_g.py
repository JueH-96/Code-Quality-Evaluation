import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	arr = list(map(int, data[1:1+n]))
	arr.sort()
	
	if n <= 2:
		print(0)
		return
		
	d0 = arr[1] - arr[0]
	is_ap = True
	for i in range(1, n-1):
		if arr[i+1] - arr[i] != d0:
			is_ap = False
			break
			
	if is_ap:
		m = (n - 1) // 2
		count = m * n - m * (m + 1)
		print(count)
	else:
		count = 0
		for j in range(1, n-1):
			i_ptr = j - 1
			k_ptr = j + 1
			while i_ptr >= 0 and k_ptr < n:
				diff_left = arr[j] - arr[i_ptr]
				diff_right = arr[k_ptr] - arr[j]
				if diff_left == diff_right:
					count += 1
					i_ptr -= 1
					k_ptr += 1
				elif diff_left < diff_right:
					i_ptr -= 1
				else:
					k_ptr += 1
		print(count)

if __name__ == "__main__":
	main()