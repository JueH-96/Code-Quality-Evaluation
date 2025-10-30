def min_distance(start, end, forbidden, M):
    # Calculate distances
    clockwise = (end - start + M) % M
    counterclockwise = (start - end + M) % M
    
    # Check if clockwise path is blocked
    blocked_clockwise = any((start < x < end) if start < end else (x > start or x < end) for x in forbidden)
    # Check if counterclockwise path is blocked
    blocked_counterclockwise = any((end < x < start) if end < start else (x < start or x > end) for x in forbidden)
    
    if not blocked_clockwise and not blocked_counterclockwise:
        return min(clockwise, counterclockwise)
    elif blocked_clockwise:
        if not blocked_counterclockwise:
            return counterclockwise
        else:
            return -1
    elif blocked_counterclockwise:
        return clockwise
    else:
        return -1

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+2*N]))
    
    total_ops = 0
    for i in range(N):
        if i == 0:
            forbidden = [A[1]]
        elif i == N-1:
            forbidden = [A[N-2]]
        else:
            forbidden = [A[i-1], A[i+1]]
        
        dist = min_distance(A[i], B[i], forbidden, M)
        if dist == -1:
            print(-1)
            return
        total_ops += dist
    
    print(total_ops)

if __name__ == '__main__':
    main()