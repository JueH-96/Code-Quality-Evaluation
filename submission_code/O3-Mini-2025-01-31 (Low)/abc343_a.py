def main():
    import sys
    data = sys.stdin.read().split()
    A = int(data[0])
    B = int(data[1])
    sum_ab = A + B
    # If sum_ab is not 0, we can print 0; otherwise print 1.
    if sum_ab != 0:
        print(0)
    else:
        print(1)

if __name__ == '__main__':
    main()