def main():
	data = input().split()
	A = int(data[0])
	B = int(data[1])
	total = A + B

	if total == 0:
		print(9)
	elif total % 2 == 0:
		print(total // 2)
	else:
		if A != total:
			print(A)
		else:
			print(B)

if __name__ == '__main__':
	main()