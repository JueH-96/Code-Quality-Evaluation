import bisect

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    P = int(data[2])
    A = list(map(int, data[3:3+n]))
    B = list(map(int, data[3+n:3+n+m]))
    
    B.sort()
    prefix = [0]
    for b in B:
        prefix.append(prefix[-1] + b)
    
    total_pairs = n * m
    total_below = 0
    F = 0
    
    for a in A:
        x = P - a
        idx = bisect.bisect_left(B, x)
        count_i = idx
        sum_i = prefix[idx]
        F += (a * count_i + sum_i)
        total_below += count_i
    
    G = (total_pairs - total_below) * P
    answer = F + G
    print(answer)

if __name__ == "__main__":
    main()