import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(-1)
		return
	
	n = int(data[0])
	X = int(data[1])
	A = list(map(int, data[2:2+n]))
	
	arr = []
	for i in range(n):
		arr.append((A[i], i+1))
	
	arr.sort(key=lambda x: x[0])
	
	for i in range(n-2):
		if i > 0 and arr[i][0] == arr[i-1][0]:
			continue
			
		if arr[i][0] + arr[i+1][0] + arr[i+2][0] > X:
			break
			
		if arr[i][0] + arr[n-2][0] + arr[n-1][0] < X:
			continue
			
		left = i+1
		right = n-1
		while left < right:
			total = arr[i][0] + arr[left][0] + arr[right][0]
			if total == X:
				indices = [arr[i][1], arr[left][1], arr[right][1]]
				indices.sort()
				print(f"{indices[0]} {indices[1]} {indices[2]}")
				return
			elif total < X:
				left += 1
				while left < right and arr[left][0] == arr[left-1][0]:
					left += 1
			else:
				right -= 1
				while left < right and arr[right][0] == arr[right+1][0]:
					right -= 1
					
	print(-1)

if __name__ == "__main__":
	main()