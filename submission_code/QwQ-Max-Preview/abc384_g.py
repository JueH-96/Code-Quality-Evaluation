import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N]))
    ptr += N
    K = int(input[ptr])
    ptr += 1
    queries = []
    for _ in range(K):
        X = int(input[ptr])
        Y = int(input[ptr+1])
        queries.append((X, Y))
        ptr += 2
    
    for X, Y in queries:
        sorted_A = sorted(A[:X])
        sorted_B = sorted(B[:Y])
        sum_B = sum(sorted_B)
        prefix = [0]
        for b in sorted_B:
            prefix.append(prefix[-1] + b)
        total = 0
        for a in sorted_A:
            cnt_le = bisect.bisect_right(sorted_B, a)
            sum_le = prefix[cnt_le]
            sum_gt = sum_B - sum_le
            cnt_gt = Y - cnt_le
            contribution = a * cnt_le - sum_le + (sum_gt - a * cnt_gt)
            total += contribution
        print(total)
        
if __name__ == "__main__":
    main()