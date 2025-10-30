import sys

def main() -> None:
    # Read N
    N_line = sys.stdin.readline().strip()
    # Ensure it is not empty in case of trailing newlines
    while N_line == '':
        N_line = sys.stdin.readline().strip()
    N = int(N_line)

    # Build result string
    res_chars = []
    for i in range(1, N + 1):
        if i % 3 == 0:
            res_chars.append('x')
        else:
            res_chars.append('o')

    # Output
    print(''.join(res_chars))

if __name__ == "__main__":
    main()