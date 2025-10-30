def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    numbers = list(map(int, input_data[1:]))
    
    # Check if all numbers are equal using set conversion.
    if len(set(numbers)) == 1:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()