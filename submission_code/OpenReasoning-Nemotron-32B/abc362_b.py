def main():
	x0, y0 = map(int, input().split())
	x1, y1 = map(int, input().split())
	x2, y2 = map(int, input().split())
	
	# Vector AB and AC from A
	ab_x = x1 - x0
	ab_y = y1 - y0
	ac_x = x2 - x0
	ac_y = y2 - y0
	dotA = ab_x * ac_x + ab_y * ac_y
	
	# Vector BA and BC from B
	ba_x = x0 - x1
	ba_y = y0 - y1
	bc_x = x2 - x1
	bc_y = y2 - y1
	dotB = ba_x * bc_x + ba_y * bc_y
	
	# Vector CA and CB from C
	ca_x = x0 - x2
	ca_y = y0 - y2
	cb_x = x1 - x2
	cb_y = y1 - y2
	dotC = ca_x * cb_x + ca_y * cb_y
	
	if dotA == 0 or dotB == 0 or dotC == 0:
		print("Yes")
	else:
		print("No")

if __name__ == "__main__":
	main()