def main():
    import sys
    input = sys.stdin.readline
    
    # Read inputs
    N, M, P = map(int, input().split())
    
    # If the first full moon day is beyond day N, answer is 0.
    if M > N:
        print(0)
        return
    
    # Otherwise, the days on which full moons occur starting at M are:
    # M, M+P, M+2P, ..., up to <= N.
    # The count is 1 + (N - M) // P.
    answer = 1 + (N - M) // P
    print(answer)

if __name__ == '__main__':
    main()