def main():
	import sys
	data = sys.stdin.read().split()
	V1 = int(data[0])
	V2 = int(data[1])
	V3 = int(data[2])
	
	total_union = V1 + V2 + V3
	temp = 1029 - V1 - 2 * V2 - V3
	if temp < 0 or temp % 2 != 0:
		print("No")
		return
		
	I123_val = temp // 2
	if I123_val > 343:
		print("No")
		return
		
	sum_pair = V2 + 3 * I123_val
	
	for dx1 in range(-7, 8):
		for dy1 in range(-7, 8):
			for dz1 in range(-7, 8):
				for dx2 in range(-7, 8):
					for dy2 in range(-7, 8):
						for dz2 in range(-7, 8):
							gaps_x = [abs(dx1), abs(dx2), abs(dx1 - dx2)]
							gap_x = max(gaps_x)
							triple_x = 7 - gap_x if gap_x < 7 else 0
							
							gaps_y = [abs(dy1), abs(dy2), abs(dy1 - dy2)]
							gap_y = max(gaps_y)
							triple_y = 7 - gap_y if gap_y < 7 else 0
							
							gaps_z = [abs(dz1), abs(dz2), abs(dz1 - dz2)]
							gap_z = max(gaps_z)
							triple_z = 7 - gap_z if gap_z < 7 else 0
							
							I123_candidate = triple_x * triple_y * triple_z
							if I123_candidate != I123_val:
								continue
							
							I12 = (7 - abs(dx1) if abs(dx1) < 7 else 0) * (7 - abs(dy1) if abs(dy1) < 7 else 0) * (7 - abs(dz1) if abs(dz1) < 7 else 0)
							I13 = (7 - abs(dx2) if abs(dx2) < 7 else 0) * (7 - abs(dy2) if abs(dy2) < 7 else 0) * (7 - abs(dz2) if abs(dz2) < 7 else 0)
							I23 = (7 - abs(dx1 - dx2) if abs(dx1 - dx2) < 7 else 0) * (7 - abs(dy1 - dy2) if abs(dy1 - dy2) < 7 else 0) * (7 - abs(dz1 - dz2) if abs(dz1 - dz2) < 7 else 0)
							
							if I12 + I13 + I23 == sum_pair:
								print("Yes")
								print(f"0 0 0 {dx1} {dy1} {dz1} {dx2} {dy2} {dz2}")
								return
								
	print("No")

if __name__ == "__main__":
	main()