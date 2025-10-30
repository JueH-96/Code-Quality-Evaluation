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
    prefix = [0] * (m + 1)
    for i in range(1, m + 1):
        prefix[i] = prefix[i-1] + B[i-1]
    
    total = 0
    for a in A:
        x = P - a
        j0 = bisect.bisect_left(B, x)
        count1 = j0
        count2 = m - j0
        total += a * count1 + prefix[j0] + P * count2
    
    print(total)

if __name__ == "__main__":
    main()