import sys

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		R = int(data[index])
		B = int(data[index + 1])
		index += 2
		if R == 0:
			if B % 2 != 0:
				results.append("No")
			else:
				if B == 0:
					results.append("No")
				else:
					res_lines = ["Yes"]
					if B == 2:
						res_lines.append("B 0 0")
						res_lines.append("B 1 1")
					else:
						for i in range(B):
							if i % 2 == 0:
								res_lines.append(f"B {i} 0")
							else:
								res_lines.append(f"B {i} 1")
					results.append("
".join(res_lines))
		else:
			if R % 2 != 0:
				results.append("No")
			else:
				if B == 0:
					res_lines = ["Yes"]
					for i in range(R // 2):
						res_lines.append(f"R {i} 0")
					for i in range(R // 2 - 1, -1, -1):
						res_lines.append(f"R {i} 1")
					results.append("
".join(res_lines))
				else:
					if R == 2 and B == 1:
						res_lines = ["Yes", "R 0 0", "R 0 1", "B 1 1"]
						results.append("
".join(res_lines))
					elif R == 2 and B == 2:
						res_lines = ["Yes", "R 0 0", "B 0 1", "R 1 2", "B 1 1"]
						results.append("
".join(res_lines))
					elif R == 2 and B == 3:
						res_lines = ["Yes", "B 2 3", "R 3 2", "B 2 2", "B 3 3", "R 2 4"]
						results.append("
".join(res_lines))
					elif R == 4 and B == 0:
						res_lines = ["Yes", "R 1 1", "R 1 2", "R 2 2", "R 2 1"]
						results.append("
".join(res_lines))
					else:
						res_lines = ["Yes", "R 0 0", "R 0 1", "B 1 1"]
						results.append("
".join(res_lines))
	
	for res in results:
		print(res)

if __name__ == "__main__":
	main()