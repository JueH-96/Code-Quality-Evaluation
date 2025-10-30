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
    
    B_sorted = sorted(B)
    
    prefix_sum = [0] * (M + 1)
    for i in range(M):
        prefix_sum[i + 1] = prefix_sum[i] + B_sorted[i]
    
    total = 0
    for a in A:
        threshold = P - a
        cnt = bisect.bisect_right(B_sorted, threshold)
        sum_part = a * cnt + prefix_sum[cnt]
        other_part = P * (M - cnt)
        total += sum_part + other_part
    
    print(total)

if __name__ == "__main__":
    main()