def main():
    import sys
    input_data = sys.stdin.read().strip()
    if not input_data:
        return
    X = int(input_data)
    
    factorial = 1
    N = 1
    while factorial < X:
        N += 1
        factorial *= N
    # At this point factorial should equal X
    print(N)

if __name__ == '__main__':
    main()