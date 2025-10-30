# YOUR CODE HERE
def main():
    N = int(input().strip())
    
    # Find divisors j of N in [1..9].
    divisors = [j for j in range(1, 10) if N % j == 0]
    
    result = []
    for i in range(N + 1):
        # Default character is '-'
        s_i = '-'
        # Check each divisor in ascending order
        for j in divisors:
            # Since j divides N, N//j is integral
            if i % (N // j) == 0:
                s_i = str(j)  # Assign the smallest possible digit
                break
        result.append(s_i)
    
    # Print the result as a single string
    print("".join(result))

# Do not forget to call main() to execute the solution
if __name__ == "__main__":
    main()