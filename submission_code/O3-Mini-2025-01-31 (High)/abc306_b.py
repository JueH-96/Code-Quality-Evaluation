def main():
    import sys
    tokens = sys.stdin.read().split()
    total = 0
    # Each token corresponds to A_i for i from 0 to 63
    # Multiply each by 2**i (or use left shifting) and sum
    for i, token in enumerate(tokens):
        total += int(token) << i  # int(token) * (2**i)
    print(total)

if __name__ == '__main__':
    main()