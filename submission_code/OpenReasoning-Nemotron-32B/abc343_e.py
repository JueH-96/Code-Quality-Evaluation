def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	V1 = int(data[0])
	V2 = int(data[1])
	V3 = int(data[2])
	total = 3 * 343  # 1029

	if V1 != total - 2 * V2 - 3 * V3:
		print("No")
		return

	arrangements = [
		([0, -1, 0], [-1, 0, 0], 1, 2),
		([-1, 0, 0], [0, 0, -1], 1, 3),
		([0, -1, 0], [0, 0, -1], 2, 3)
	]
	
	found = False
	result = None
	for shifts2, shifts3, dim_a, dim_b in arrangements:
		for a in range(0, 8):
			if a == 0:
				if V3 != 0:
					continue
				if V2 % 49 != 0:
					continue
				b = V2 // 49
				if b < 0 or b > 7:
					continue
				s2 = list(shifts2)
				s3 = list(shifts3)
				if s2[dim_b - 1] == -1:
					s2[dim_b - 1] = 7 - b
				if s3[dim_a - 1] == -1:
					s3[dim_a - 1] = 7 - a
				found = True
				result = (0, 0, 0, s2[0], s2[1], s2[2], s3[0], s3[1], s3[2])
				break
			else:
				if V3 % (7 * a) != 0:
					continue
				b_val = V3 // (7 * a)
				if b_val < 0 or b_val > 7:
					continue
				if 49 * (a + b_val) == V2 + 2 * V3:
					s2 = list(shifts2)
					s3 = list(shifts3)
					if s2[dim_b - 1] == -1:
						s2[dim_b - 1] = 7 - b_val
					if s3[dim_a - 1] == -1:
						s3[dim_a - 1] = 7 - a
					found = True
					result = (0, 0, 0, s2[0], s2[1], s2[2], s3[0], s3[1], s3[2])
					break
		if found:
			break

	if found:
		print("Yes")
		a1, b1, c1, a2, b2, c2, a3, b3, c3 = result
		print(f"{a1} {b1} {c1} {a2} {b2} {c2} {a3} {b3} {c3}")
	else:
		print("No")

if __name__ == "__main__":
	main()