def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	k = int(data[1])
	arr = list(map(int, data[2:2+n]))
	
	arr.sort()
	need = n - k
	
	lo = 0
	hi = arr[-1] - arr[0]
	
	while lo < hi:
		mid = (lo + hi) // 2
		r = 0
		found = False
		for l in range(n):
			while r < n and arr[r] - arr[l] <= mid:
				r += 1
			if r - l >= need:
				found = True
				break
		if found:
			hi = mid
		else:
			lo = mid + 1
			
	print(lo)

if __name__ == "__main__":
	main()