def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    S = input_data[1].strip()
    
    total = 0
    dp = 0  # dp for substrings ending at current index
    # Process digits based on 1-index logic, but adjust python indices.
    for i, digit in enumerate(S, 1):
        dp = dp * 10 + int(digit) * i
        total += dp

    sys.stdout.write(str(total))
    
if __name__ == '__main__':
    main()