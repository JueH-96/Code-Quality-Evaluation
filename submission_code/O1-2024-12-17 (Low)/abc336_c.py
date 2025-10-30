def main():
    import sys
    N = int(sys.stdin.readline().strip())
    
    # If N=1, the answer is 0
    if N == 1:
        print(0)
        return
    
    # We will convert (N - 1) to base 5, and map each digit:
    # 0 -> 0, 1 -> 2, 2 -> 4, 3 -> 6, 4 -> 8
    # This gives the N-th smallest good integer.
    convert_map = ['0', '2', '4', '6', '8']
    x = N - 1
    result_digits = []
    
    while x > 0:
        d = x % 5
        x //= 5
        result_digits.append(convert_map[d])
    
    # Digits are collected in reverse order, so reverse them and print
    print(''.join(result_digits[::-1]))

# Do not forget to call main() at the end!
if __name__ == "__main__":
    main()