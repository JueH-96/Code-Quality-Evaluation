import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print("No")
		return
	
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	if all(x == 0 for x in A):
		if n % 4 == 0:
			print("Yes")
		else:
			print("No")
		return
			
	i = 0
	while i < n and A[i] != 1:
		i += 1
		
	segments = []
	count = 0
	j = (i + 1) % n
	current = j
	while current != i:
		if A[current] == 0:
			count += 1
		else:
			if count > 0:
				segments.append(count)
				count = 0
		current = (current + 1) % n
		
	if count > 0:
		segments.append(count)
		
	for L in segments:
		if L == 1 or L == 2:
			continue
		elif L % 2 == 1:
			print("No")
			return
		else:
			continue
			
	print("Yes")

if __name__ == "__main__":
	main()