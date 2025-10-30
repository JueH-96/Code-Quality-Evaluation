import sys
import bisect

def main():
	data = sys.stdin.read().splitlines()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		n = int(data[index]); index += 1
		A = data[index].strip(); index += 1
		B = data[index].strip(); index += 1
		
		P = []
		Q = []
		for i in range(n):
			if A[i] == '1':
				P.append(i)
			if B[i] == '1':
				Q.append(i)
				
		if len(P) < len(Q):
			results.append("-1")
			continue
			
		low = 0
		high = n
		ans_d = n
		
		while low <= high:
			mid = (low + high) // 2
			valid_flag = True
			j = 0
			for y in Q:
				while j < len(P) and P[j] < y - mid:
					j += 1
				if j >= len(P) or P[j] > y + mid:
					valid_flag = False
					break
			if not valid_flag:
				low = mid + 1
				continue
				
			j = 0
			for x in P:
				while j < len(Q) and Q[j] < x - mid:
					j += 1
				if j >= len(Q) or Q[j] > x + mid:
					valid_flag = False
					break
					
			if valid_flag:
				ans_d = mid
				high = mid - 1
			else:
				low = mid + 1
				
		results.append(str(ans_d))
		
	print("
".join(results))

if __name__ == '__main__':
	main()