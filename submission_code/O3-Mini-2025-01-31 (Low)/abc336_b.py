def main():
    import sys
    input_data = sys.stdin.read().strip()
    if not input_data:
        return
    N = int(input_data)
    count = 0
    # Count the trailing zeros in binary representation (i.e., factors of 2)
    while N % 2 == 0:
        count += 1
        N //= 2
    print(count)

if __name__ == "__main__":
    main()