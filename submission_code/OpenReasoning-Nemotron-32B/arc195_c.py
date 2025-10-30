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
		n = R + B
		if R == 0:
			if B % 2 != 0:
				results.append("No")
			else:
				if B == 0:
					results.append("No")
				else:
					results.append("Yes")
					x, y = 0, 0
					out_lines = []
					out_lines.append(f"B {x + 1} {y + 1}")
					for i in range(1, B):
						if i % 4 == 1:
							x += 1
							y += 1
						elif i % 4 == 2:
							x -= 1
							y += 1
						elif i % 4 == 3:
							x -= 1
							y -= 1
						else:
							x += 1
							y -= 1
						out_lines.append(f"B {x + 1} {y + 1}")
					results.extend(out_lines)
		else:
			if R % 2 != 0:
				results.append("No")
			else:
				if B == 0:
					if R == 2:
						results.append("Yes")
						results.append("R 1 1")
						results.append("R 2 1")
					else:
						results.append("Yes")
						out_lines = []
						for i in range(R // 2):
							out_lines.append(f"R 1 {i + 1}")
						for i in range(R // 2 - 1, -1, -1):
							out_lines.append(f"R 2 {i + 1}")
						results.extend(out_lines)
				else:
					if R == 2 and B == 3:
						results.append("Yes")
						results.append("B 2 3")
						results.append("R 3 2")
						results.append("B 2 2")
						results.append("B 3 3")
						results.append("R 2 4")
					elif R == 2 and B == 1:
						results.append("Yes")
						results.append("B 1 1")
						results.append("R 2 2")
						results.append("R 1 2")
					elif R == 2 and B == 2:
						results.append("Yes")
						results.append("B 1 1")
						results.append("R 2 2")
						results.append("B 1 2")
						results.append("R 2 1")
					else:
						results.append("No")
	for res in results:
		print(res)

if __name__ == "__main__":
	main()