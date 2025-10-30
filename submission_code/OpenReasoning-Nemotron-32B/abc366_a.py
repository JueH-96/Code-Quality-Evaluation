def main():
	data = input().split()
	N = int(data[0])
	T = int(data[1])
	A = int(data[2])
	R = N - T - A
	if T > A + R or A > T + R:
		print("Yes")
	else:
		print("No")

if __name__ == '__main__':
	main()