def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    sum_a = 0
    max_delta = -float('inf')
    idx = 1
    for _ in range(n):
        a = int(input[idx])
        b = int(input[idx+1])
        idx +=2
        sum_a += a
        delta = b - a
        if delta > max_delta:
            max_delta = delta
    print(sum_a + max_delta)
    
if __name__ == '__main__':
    main()