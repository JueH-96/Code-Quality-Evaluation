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
	P = [0] * (n+1)
	for i in range(1, n+1):
		P[i] = P[i-1] + A[i-1]
		
	seen = set()
	seen.add(0)
	for j in range(n):
		current = P[j+1]
		if (current - S_val) in seen:
			print("Yes")
			return
		seen.add(current)
		
	all_P = set(P[1:])
	
	if T == 0:
		print("No")
		return
		
	m_min = max(1, (S_val - 1) // T)
	m_max = (S_val + T) // T
	
	if m_min > m_max:
		print("No")
		return
		
	for m in range(m_min, m_max + 1):
		base = S_val - m * T
		for i in range(n):
			candidate = base + P[i]
			if candidate in all_P:
				print("Yes")
				return
				
	print("No")

if __name__ == "__main__":
	main()