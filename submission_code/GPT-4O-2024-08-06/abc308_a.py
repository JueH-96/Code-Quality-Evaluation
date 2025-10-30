# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = list(map(int, input().strip().split()))
    
    # Check if the sequence is monotonically non-decreasing
    is_non_decreasing = all(data[i] <= data[i + 1] for i in range(7))
    
    # Check if all numbers are between 100 and 675 inclusive
    in_range = all(100 <= num <= 675 for num in data)
    
    # Check if all numbers are multiples of 25
    multiples_of_25 = all(num % 25 == 0 for num in data)
    
    # If all conditions are satisfied, print "Yes", otherwise print "No"
    if is_non_decreasing and in_range and multiples_of_25:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()