import sys

def main():
	data = sys.stdin.read().splitlines()
	if len(data) < 2:
		print(0)
		return
	
	parts = data[0].split()
	if not parts:
		print(0)
		return
		
	try:
		N = int(parts[0])
		M = int(parts[1])
	except:
		print(0)
		return
		
	S = data[1].strip()
	
	for x in range(0, N+1):
		plain = M
		logo = x
		for c in S:
			if c == '0':
				plain = M
				logo = x
			elif c == '1':
				if plain > 0:
					plain -= 1
				elif logo > 0:
					logo -= 1
				else:
					break
			elif c == '2':
				if logo > 0:
					logo -= 1
				else:
					break
		else:
			print(x)
			return

if __name__ == "__main__":
	main()