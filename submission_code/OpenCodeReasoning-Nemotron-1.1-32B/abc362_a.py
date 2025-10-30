def main():
	R, G, B = map(int, input().split())
	C = input().strip()
	
	if C == "Red":
		ans = min(G, B)
	elif C == "Green":
		ans = min(R, B)
	else:
		ans = min(R, G)
		
	print(ans)

if __name__ == "__main__":
	main()