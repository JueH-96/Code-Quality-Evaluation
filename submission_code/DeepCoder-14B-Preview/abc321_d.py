import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    P = int(data[idx+2])
    idx +=3
    
    A = list(map(int, data[idx:idx+N]))
    idx += N
    B = list(map(int, data[idx:idx+M]))
    
    B.sort()
    
    # Compute prefix sums for B
    prefix = [0]
    current = 0
    for b in B:
        current += b
        prefix.append(current)
    
    sum1 = 0
    total_count = 0
    
    for a in A:
        target = P - a
        cnt = bisect.bisect_right(B, target)
        sum_b = prefix[cnt]
        sum1 += a * cnt + sum_b
        total_count += cnt
    
    remaining = (N * M - total_count) * P
    total = sum1 + remaining
    print(total)

if __name__ == "__main__":
    main()