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
    idx +=N
    B = list(map(int, data[idx:idx+M]))
    
    B.sort()
    prefix = [0] * (M +1)
    for i in range(M):
        prefix[i+1] = prefix[i] + B[i]
    
    total = 0
    for a in A:
        t = P - a
        idx_b = bisect.bisect_left(B, t)
        sum_less = prefix[idx_b]
        count_ge = M - idx_b
        contribution = count_ge * P + (a * idx_b + sum_less)
        total += contribution
    print(total)

if __name__ == "__main__":
    main()