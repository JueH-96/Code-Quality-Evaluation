def main():
    import sys
    input_data = sys.stdin.read().split()
    X = int(input_data[0])
    
    product = 1
    n = 1
    while product < X:
        n += 1
        product *= n
        if product == X:
            print(n)
            return

if __name__ == '__main__':
    main()