def main():
    import sys
    input_data = sys.stdin.read().split()
    n = int(input_data[0])
    # Next n integers are A, following n integers are B.
    A = list(map(int, input_data[1:1+n]))
    B = list(map(int, input_data[1+n:1+2*n]))
    
    max_A = max(A)
    max_B = max(B)
    
    print(max_A + max_B)

if __name__ == '__main__':
    main()