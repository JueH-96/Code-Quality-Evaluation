import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	x = int(data[1])
	y = int(data[2])
	s = data[3]
	t = data[4]
	
	if s == t:
		print("Yes")
		return
		
	count0_s = s.count('0')
	count0_t = t.count('0')
	if count0_s != count0_t:
		print("No")
		return
		
	f = [0] * (n + 10)
	current = 0
	for i in range(n):
		current += f[i]
		effective_s = (ord(s[i]) - ord('0') + current) % 2
		target = ord(t[i]) - ord('0')
		if effective_s != target:
			if i + x > n:
				print("No")
				return
			current += 1
			if i + x < n + 10:
				f[i + x] -= 1
			if i + x + y < n + 10:
				f[i + x + y] -= 1
				
	print("Yes")

if __name__ == '__main__':
	main()