def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    A = int(input_data[0])
    B = int(input_data[1])
    s = A + B
    # If s is not 0, we can safely print 0.
    # If s is 0, printing 1 is valid.
    if s != 0:
        print(0)
    else:
        print(1)

if __name__ == '__main__':
    main()