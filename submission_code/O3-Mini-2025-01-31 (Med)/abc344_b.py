def main():
    import sys
    data = sys.stdin.read().splitlines()
    # Convert each line to an integer
    numbers = [int(line) for line in data if line.strip() != ""]
    # Print in reverse order
    for num in reversed(numbers):
        print(num)

if __name__ == '__main__':
    main()