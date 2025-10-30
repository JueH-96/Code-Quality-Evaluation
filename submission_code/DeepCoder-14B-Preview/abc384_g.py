import bisect
from itertools import accumulate

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    
    n = int(data[ptr])
    ptr += 1
    
    A = list(map(int, data[ptr:ptr+n]))
    ptr += n
    
    B = list(map(int, data[ptr:ptr+n]))
    ptr += n
    
    k = int(data[ptr])
    ptr += 1
    
    queries = []
    for _ in range(k):
        X = int(data[ptr])
        Y = int(data[ptr+1])
        queries.append((X, Y))
        ptr += 2
    
    for X, Y in queries:
        A_sub = A[:X]
        A_sub.sort()
        B_sub = B[:Y]
        B_sub.sort()
        
        pre_b = list(accumulate(B_sub, initial=0))
        sum_b = pre_b[-1]
        
        total = 0
        for a in A_sub:
            pos = bisect.bisect_right(B_sub, a)
            sum_less = pre_b[pos]
            sum_more = sum_b - sum_less
            cnt_less = pos
            cnt_more = Y - cnt_less
            contribution = (a * cnt_less - sum_less) + (sum_more - a * cnt_more)
            total += contribution
        
        print(total)

if __name__ == '__main__':
    main()