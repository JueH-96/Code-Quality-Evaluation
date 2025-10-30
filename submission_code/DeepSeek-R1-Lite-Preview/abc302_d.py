import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    D = int(data[2])
    
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+N+M]))
    
    A.sort()
    B.sort()
    
    max_sum = -1
    
    for i in range(N-1, -1, -1):
        a = A[i]
        upper = a + D
        pos = bisect.bisect_right(B, upper) - 1
        if pos >= 0 and B[pos] >= a - D:
            current_sum = a + B[pos]
            if current_sum > max_sum:
                max_sum = current_sum
    
    print(max_sum)

if __name__ == '__main__':
    main()