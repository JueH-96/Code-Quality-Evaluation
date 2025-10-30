# YOUR CODE HERE
import sys

def solve():
    import sys
    import bisect
    from collections import defaultdict
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx]); idx +=1
    results = []
    for _ in range(T):
        N = int(data[idx]); K = int(data[idx+1]); idx +=2
        A = list(map(int, data[idx:idx+N])); idx +=N
        B = list(map(int, data[idx:idx+N])); idx +=N
        a_map = defaultdict(list)
        b_map = defaultdict(list)
        for pos, val in enumerate(A,1):
            a_map[val].append(pos)
        for pos, val in enumerate(B,1):
            b_map[val].append(pos)
        possible = True
        for v in b_map:
            if v not in a_map:
                possible = False
                break
            S = sorted(a_map[v])
            T_v = sorted(b_map[v])
            intervals = []
            for s in S:
                start = max(1, s - K)
                end = min(N, s + K)
                intervals.append((start, end))
            intervals.sort()
            merged = []
            for interval in intervals:
                if not merged or interval[0] > merged[-1][1] +1:
                    merged.append(list(interval))
                else:
                    merged[-1][1] = max(merged[-1][1], interval[1])
            # Now check if all T_v are covered by merged intervals
            m = 0
            len_m = len(merged)
            for t in T_v:
                while m < len_m and merged[m][1] < t:
                    m +=1
                if m == len_m or merged[m][0] > t:
                    possible = False
                    break
            if not possible:
                break
        results.append("Yes" if possible else "No")
    print('
'.join(results))