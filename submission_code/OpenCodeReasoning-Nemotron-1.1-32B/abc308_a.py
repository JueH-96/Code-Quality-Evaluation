def main():
	data = list(map(int, input().split()))
	
	# Check if the sequence is non-decreasing
	is_non_decreasing = True
	for i in range(7):
		if data[i] > data[i+1]:
			is_non_decreasing = False
			break
			
	# Check if all numbers are between 100 and 675 inclusive
	in_range = True
	for x in data:
		if x < 100 or x > 675:
			in_range = False
			break
			
	# Check if all numbers are multiples of 25
	multiples_of_25 = True
	for x in data:
		if x % 25 != 0:
			multiples_of_25 = False
			break
			
	if is_non_decreasing and in_range and multiples_of_25:
		print("Yes")
	else:
		print("No")

if __name__ == "__main__":
	main()