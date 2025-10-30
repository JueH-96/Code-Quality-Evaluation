def main():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    
    # Find divisors of N between 1 and 9
    divisors = [j for j in range(1, 10) if N % j == 0]
    
    # Prepare the result string
    result = []
    
    for i in range(N + 1):
        # Check for the smallest divisor j such that i is a multiple of N/j
        valid_divisors = [j for j in divisors if i % (N // j) == 0]
        if valid_divisors:
            result.append(str(min(valid_divisors)))
        else:
            result.append('-')
    
    # Join the result list into a string and print it
    print(''.join(result))

if __name__ == "__main__":
    main()