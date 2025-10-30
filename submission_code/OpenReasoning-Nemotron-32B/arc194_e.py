import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print("No")
		return
	n, x, y = map(int, data[0].split())
	s = data[1].strip()
	t = data[2].strip()
	
	if s == t:
		print("Yes")
		return
		
	count_s = s.count('1')
	count_t = t.count('1')
	if count_s != count_t:
		print("No")
		return
		
	L = x + y
	diff = [0] * (n + L + 10)
	c = 0
	for i in range(n):
		c ^= diff[i]
		current_bit = 1 if s[i] == '1' else 0
		current_bit ^= c
		target_bit = 1 if t[i] == '1' else 0
		
		if current_bit == target_bit:
			continue
			
		if i + L > n:
			print("No")
			return
			
		c ^= 1
		if i + x < len(diff):
			diff[i + x] ^= 1
		if i + L < len(diff):
			diff[i + L] ^= 1
			
	print("Yes")

if __name__ == '__main__':
	main()