import sys

def main() -> None:
    # Read N
    N_line = sys.stdin.readline().strip()
    if not N_line:
        return
    N = int(N_line)

    # Convert (N-1) to base-5 and map the digits
    k = N - 1
    if k == 0:
        print(0)
        return

    digits = []
    while k:
        digits.append(k % 5)
        k //= 5

    # Map 0→0, 1→2, 2→4, 3→6, 4→8 and build the answer
    even_number_str = ''.join(str(d * 2) for d in reversed(digits))
    print(even_number_str)

if __name__ == "__main__":
    main()