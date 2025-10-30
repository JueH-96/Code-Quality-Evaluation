def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    A = int(input_data[0])
    B = int(input_data[1])
    result = A ** B + B ** A
    print(result)

if __name__ == '__main__':
    main()