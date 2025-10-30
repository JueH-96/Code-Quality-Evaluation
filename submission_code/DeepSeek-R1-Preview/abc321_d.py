import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    P = int(input[ptr])
    ptr += 1
    
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+M]))
    ptr += M
    
    B.sort()
    
    pre_sum = [0]
    current = 0
    for b in B:
        current += b
        pre_sum.append(current)
    
    total = 0
    for a in A:
        x = P - a
        cnt = bisect.bisect_right(B, x)
        sum_b = pre_sum[cnt]
        contribution = a * cnt + sum_b + (M - cnt) * P
        total += contribution
    
    print(total)

if __name__ == '__main__':
    main()