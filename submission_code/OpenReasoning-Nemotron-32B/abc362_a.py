def main():
	R, G, B = map(int, input().split())
	C = input().strip()
	
	allowed_costs = []
	if C != "Red":
		allowed_costs.append(R)
	if C != "Green":
		allowed_costs.append(G)
	if C != "Blue":
		allowed_costs.append(B)
		
	print(min(allowed_costs))

if __name__ == "__main__":
	main()