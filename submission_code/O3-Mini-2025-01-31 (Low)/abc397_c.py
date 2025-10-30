def main():
    import sys
    input_data = sys.stdin.read().split()
    n = int(input_data[0])
    A = list(map(int, input_data[1:]))
    
    prefix = [0] * n
    seen = set()
    for i in range(n):
        seen.add(A[i])
        prefix[i] = len(seen)
    
    suffix = [0] * n
    seen = set()
    for i in range(n - 1, -1, -1):
        seen.add(A[i])
        suffix[i] = len(seen)

    res = 0
    for i in range(n - 1):
        res = max(res, prefix[i] + suffix[i + 1])
    
    sys.stdout.write(str(res))
    
if __name__ == '__main__':
    main()