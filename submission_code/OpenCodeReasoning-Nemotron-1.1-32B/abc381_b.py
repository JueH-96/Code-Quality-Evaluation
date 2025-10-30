from collections import Counter

def main():
	s = input().strip()
	n = len(s)
	
	if n % 2 != 0:
		print("No")
		return
		
	for i in range(0, n, 2):
		if s[i] != s[i+1]:
			print("No")
			return
			
	freq = Counter(s)
	for count in freq.values():
		if count != 2:
			print("No")
			return
			
	print("Yes")

if __name__ == "__main__":
	main()