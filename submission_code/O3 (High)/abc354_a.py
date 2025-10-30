def main():
    import sys

    # Read Takahashi's height
    H_line = sys.stdin.readline().strip()
    if not H_line:
        return
    H = int(H_line)

    # The plant's height every morning of day d is (2^d - 1).
    # We need the smallest d such that 2^d - 1 > H  ⇔  2^d > H + 1.
    # For any positive integer N, the smallest d with 2^d > N
    # equals the bit length of N in binary.
    answer = (H + 1).bit_length()

    # Output the answer
    print(answer)

if __name__ == "__main__":
    main()