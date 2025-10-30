def main():
    import sys
    input = sys.stdin.readline

    # read input values
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    
    # sort the gift coordinates
    A.sort()
    
    # two pointers sliding window approach
    max_gifts = 0
    j = 0
    for i in range(N):
        # expand j while the next gift is within [A[i], A[i] + M)
        while j < N and A[j] < A[i] + M:
            j += 1
        # j is the first position that does not satisfy the interval inclusion
        max_gifts = max(max_gifts, j - i)
    
    print(max_gifts)

if __name__ == '__main__':
    main()