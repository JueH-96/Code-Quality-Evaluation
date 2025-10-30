# YOUR CODE HERE
import sys
import sys
def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N_Q = sys.stdin.readline().split()
    while len(N_Q) < 2:
        N_Q += sys.stdin.readline().split()
    N, Q = map(int, N_Q)
    x_input = []
    while len(x_input) < Q:
        x_input += sys.stdin.readline().split()
    x_list = list(map(int, x_input[:Q]))
    
    N = int(N)
    Q = int(Q)
    list_intervals = [[] for _ in range(N+1)]
    start = [None]*(N+1)
    size_S = [0]*Q
    S_present = [False]*(N+1)
    S_size = 0
    for t in range(1, Q+1):
        x = int(x_list[t-1])
        if S_present[x]:
            list_intervals[x].append( (start[x], t) )
            S_present[x] = False
            start[x] = None
            S_size -=1
        else:
            S_present[x] = True
            start[x] = t
            S_size +=1
        size_S[t-1] = S_size
    for x in range(1, N+1):
        if S_present[x]:
            list_intervals[x].append( (start[x], Q+1) )
    pre_sum = [0]*(Q+1)
    for t in range(1, Q+1):
        pre_sum[t] = pre_sum[t-1] + size_S[t-1]
    A = [0]*N
    for j in range(1, N+1):
        for (l, r) in list_intervals[j]:
            A[j-1] += pre_sum[r-1] - pre_sum[l-1]
    print(' '.join(map(str, A)))

if __name__ == "__main__":
    main()