import sys

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		N = int(data[index]); M = int(data[index+1]); index += 2
		boxes = []
		for i in range(N):
			v = int(data[index]); p = int(data[index+1]); index += 2
			boxes.append((v, p))
		
		boxes.sort(key=lambda x: x[0])
		ans = 0
		top = []
		
		for i in range(N-1, -1, -1):
			v, p = boxes[i]
			x = 1 - p
			
			if M >= 1:
				if M == 1:
					profit_val = v - p
				else:
					profit_val = 1 - p
				if profit_val > ans:
					ans = profit_val
					
			if M >= 2 and len(top) >= 1:
				candidate = (1 - p) + top[-1]
				if M == 2:
					candidate += (v - 1)
				if candidate > ans:
					ans = candidate
					
			if M >= 3 and len(top) >= 2:
				candidate = (1 - p) + top[-1] + top[-2]
				if M == 3:
					candidate += (v - 1)
				if candidate > ans:
					ans = candidate
					
			if M >= 4 and len(top) >= 3:
				candidate = (1 - p) + top[-1] + top[-2] + top[-3]
				if M == 4:
					candidate += (v - 1)
				if candidate > ans:
					ans = candidate
					
			new_top = top + [x]
			new_top.sort()
			if len(new_top) > 3:
				new_top = new_top[-3:]
			top = new_top
			
		results.append(str(ans))
	
	print("
".join(results))

if __name__ == "__main__":
	main()