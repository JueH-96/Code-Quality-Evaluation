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
    
    B.sort()
    A.sort(reverse=True)
    
    max_sum = -1
    
    for a in A:
        # Find the rightmost value less than or equal to a + D
        idx = bisect.bisect_right(B, a + D)
        if idx > 0:
            b = B[idx - 1]
            if b >= a - D:
                current_sum = a + b
                if current_sum > max_sum:
                    max_sum = current_sum
    
    print(max_sum if max_sum != -1 else -1)

if __name__ == '__main__':
    main()