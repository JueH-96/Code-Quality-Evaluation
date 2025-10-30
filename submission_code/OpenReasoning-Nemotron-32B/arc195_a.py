import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	A = list(map(int, data[2:2+n]))
	B = list(map(int, data[2+n:2+n+m]))
	
	left_seq = []
	j = 0
	for i in range(n):
		if j == m:
			break
		if A[i] == B[j]:
			left_seq.append(i)
			j += 1
			
	if j < m:
		print("No")
		return
		
	right_seq = []
	j = m - 1
	for i in range(n-1, -1, -1):
		if j < 0:
			break
		if A[i] == B[j]:
			right_seq.append(i)
			j -= 1
	right_seq.reverse()
	
	if left_seq == right_seq:
		print("No")
	else:
		print("Yes")

if __name__ == '__main__':
	main()