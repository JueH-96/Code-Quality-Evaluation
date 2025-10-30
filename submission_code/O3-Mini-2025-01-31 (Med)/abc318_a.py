def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N, M, P = map(int, input_data)
    
    if M > N:
        print(0)
    else:
        # Count days: M, M+P, M+2P, ... <= N
        count = (N - M) // P + 1
        print(count)

if __name__ == '__main__':
    main()