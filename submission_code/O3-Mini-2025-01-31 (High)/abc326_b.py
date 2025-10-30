def main():
    import sys
    # Read input from standard input
    inp = sys.stdin.read().strip()
    if not inp:
        return
    N = int(inp)
    
    # Loop from N to 999 to find the first 326-like number
    for num in range(N, 1000):
        hundreds = num // 100
        tens = (num // 10) % 10
        ones = num % 10
        if hundreds * tens == ones:
            print(num)
            return

# Call the main function
if __name__ == '__main__':
    main()