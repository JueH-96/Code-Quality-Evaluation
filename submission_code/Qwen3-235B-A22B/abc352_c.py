def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    sum_A = 0
    max_diff = -float('inf')
    for _ in range(N):
        A = int(input[idx])
        B = int(input[idx+1])
        idx +=2
        sum_A += A
        diff = B - A
        if diff > max_diff:
            max_diff = diff
    print(sum_A + max_diff)

if __name__ == '__main__':
    main()