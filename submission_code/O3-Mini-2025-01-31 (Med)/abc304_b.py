def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    
    if N < 10**3:
        result = N
    elif N < 10**4:
        result = (N // 10) * 10
    elif N < 10**5:
        result = (N // 100) * 100
    elif N < 10**6:
        result = (N // 1000) * 1000
    elif N < 10**7:
        result = (N // 10000) * 10000
    elif N < 10**8:
        result = (N // 100000) * 100000
    else:  # N is between 10^8 and 10^9-1 inclusive.
        result = (N // 1000000) * 1000000
        
    sys.stdout.write(str(result))
    
if __name__ == '__main__':
    main()