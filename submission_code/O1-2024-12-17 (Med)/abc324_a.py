def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    numbers = list(map(int, data[1:]))

    # Check if all numbers are equal
    if len(set(numbers)) == 1:
        print("Yes")
    else:
        print("No")

# Do not forget to call main function
main()