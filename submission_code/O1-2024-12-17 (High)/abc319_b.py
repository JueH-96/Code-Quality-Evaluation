def main():
    import sys

    data = sys.stdin.read().strip()
    N = int(data)

    # Collect divisors of N that are between 1 and 9
    divisors = [j for j in range(1, 10) if N % j == 0]

    result = []
    for i in range(N + 1):
        char = '-'
        for j in divisors:
            if i % (N // j) == 0:
                char = str(j)
                break
        result.append(char)

    print("".join(result))

# Do not forget to call main()
main()