import sys

def main():
    N_line = sys.stdin.readline().strip()
    if not N_line:
        return
    N = int(N_line)

    # Divisors of N that are between 1 and 9 (inclusive), in ascending order
    small_divisors = [j for j in range(1, 10) if N % j == 0]

    output_chars = []
    for i in range(N + 1):
        char = '-'
        # Find the smallest divisor j (if any) such that i is a multiple of N / j
        for j in small_divisors:
            if i % (N // j) == 0:
                char = str(j)
                break
        output_chars.append(char)

    print(''.join(output_chars))

if __name__ == "__main__":
    main()