def main():
	n = int(input().strip())
	gridA = [input().strip() for _ in range(n)]
	gridB = [input().strip() for _ in range(n)]
	
	for i in range(n):
		for j in range(n):
			if gridA[i][j] != gridB[i][j]:
				print(f"{i+1} {j+1}")
				return

if __name__ == "__main__":
	main()