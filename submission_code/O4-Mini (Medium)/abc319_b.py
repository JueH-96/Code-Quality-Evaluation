import sys

def main():
    data = sys.stdin.read().strip().split()
    N = int(data[0])

    # Precompute the valid divisors j of N in [1..9]
    valid_js = [j for j in range(1, 10) if N % j == 0]

    result = []
    # For each i from 0 to N, find the smallest j such that i is a multiple of N/j
    for i in range(N + 1):
        ch = '-'
        for j in valid_js:
            step = N // j
            if i % step == 0:
                ch = str(j)
                break
        result.append(ch)

    # Output the resulting string
    sys.stdout.write("".join(result))

if __name__ == "__main__":
    main()