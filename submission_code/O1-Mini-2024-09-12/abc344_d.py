import sys

def main():
    import sys
    import sys
    T = sys.stdin.readline().strip()
    N_line = sys.stdin.readline()
    while N_line.strip()=='':
        N_line = sys.stdin.readline()
    N = int(N_line)
    bags = []
    for _ in range(N):
        parts = sys.stdin.readline().strip().split()
        while len(parts)==0:
            parts = sys.stdin.readline().strip().split()
        A_i = int(parts[0])
        strings = parts[1:]
        while len(strings) < A_i:
            strings += sys.stdin.readline().strip().split()
        bags.append(strings)
    INF = float('inf')
    len_T = len(T)
    dp_prev = [INF]*(len_T+1)
    dp_prev[0] = 0
    for bag in bags:
        dp_curr = [INF]*(len_T+1)
        for j in range(len_T+1):
            if dp_prev[j] == INF:
                continue
            # Do nothing
            if dp_curr[j] > dp_prev[j]:
                dp_curr[j] = dp_prev[j]
            # Choose a string from the bag
            for s in bag:
                len_s = len(s)
                if j + len_s > len_T:
                    continue
                if T[j:j+len_s] == s:
                    if dp_curr[j+len_s] > dp_prev[j] +1:
                        dp_curr[j+len_s] = dp_prev[j]+1
        dp_prev = dp_curr
    if dp_prev[len_T] != INF:
        print(dp_prev[len_T])
    else:
        print(-1)

if __name__ == "__main__":
    main()