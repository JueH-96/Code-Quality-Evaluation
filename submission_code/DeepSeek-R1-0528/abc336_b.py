def main():
    n = int(input().strip())
    low_bit = n & -n
    print(low_bit.bit_length() - 1)

if __name__ == '__main__':
    main()