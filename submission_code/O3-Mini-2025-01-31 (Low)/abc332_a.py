def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    idx = 0
    # First three integers: N, S, K
    N = int(input_data[idx]); idx += 1
    S = int(input_data[idx]); idx += 1
    K = int(input_data[idx]); idx += 1
    total = 0
    for i in range(N):
        P = int(input_data[idx]); idx += 1
        Q = int(input_data[idx]); idx += 1
        total += P * Q
    if total < S:
        total += K
    # Output the result
    sys.stdout.write(str(total))
    
if __name__ == '__main__':
    main()