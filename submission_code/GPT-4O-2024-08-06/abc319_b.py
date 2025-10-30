# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    
    # Prepare a list of divisors of N that are between 1 and 9
    divisors = [j for j in range(1, 10) if N % j == 0]
    
    # Initialize the result string
    result = []
    
    # Iterate over each i from 0 to N
    for i in range(N + 1):
        # Find the smallest j such that i is a multiple of N/j
        found = False
        for j in divisors:
            if i % (N // j) == 0:
                result.append(str(j))
                found = True
                break
        # If no such j is found, append '-'
        if not found:
            result.append('-')
    
    # Print the result as a single string
    print(''.join(result))

if __name__ == "__main__":
    main()