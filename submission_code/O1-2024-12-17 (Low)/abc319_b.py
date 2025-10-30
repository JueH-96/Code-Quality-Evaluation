def main():
    import sys
    data = sys.stdin.read().strip()
    N = int(data)

    # Find j in [1..9] s.t. j divides N
    valid_divisors = [j for j in range(1, 10) if N % j == 0]

    # Build the result
    result = []
    for i in range(N + 1):
        char = '-'
        for j in valid_divisors:
            # Check if i is multiple of (N // j)
            if i % (N // j) == 0:
                char = str(j)
                break
        result.append(char)

    print("".join(result))

# Do not remove the next line
main()