# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_K = sys.stdin.readline().strip()
    while N_K == '':
        N_K = sys.stdin.readline().strip()
    N, K = map(int, N_K.strip().split())
    N = int(N)
    K = int(K)
    S = sys.stdin.readline().strip()
    while len(S) < N:
        S += sys.stdin.readline().strip()
    S = S[:N]
    
    N = len(S)
    l_r_list = []
    i = 0
    while i < N:
        if S[i] == '1':
            # Check if new block
            if i == 0 or S[i-1] == '0':
                l_i = i +1  # positions start from 1
            while i < N and S[i] == '1':
                i +=1
            r_i = i  # positions from l_i to r_i
            l_r_list.append( (l_i, r_i))
        else:
            i +=1
    m = len(l_r_list)
    # Ensure K <= m
    if K > m or K < 2:
        print(S)
        return

    l_prev, r_prev = l_r_list[K-2]
    l_cur, r_cur = l_r_list[K-1]

    rl = r_prev  # positions
    rk = r_cur
    lk = l_cur

    rl_index = rl -1  # indices

    rk_index = rk -1

    rk_lk_len = rk - lk +1  # length of K-th 1-block

    T = list(S)

    # Set T[ rl_index : rl_index + (rk - lk +1) ] = '1'
    end_index1 = rl_index + rk_lk_len -1

    if end_index1 >= N:
        end_index1 = N-1
    for idx in range(rl_index, end_index1 +1):
        T[idx] = '1'

    # Set T[ rl_index + (rk - lk +1) : rk_index +1 ] = '0'
    start_index2 = rl_index + rk_lk_len
    if start_index2 <= rk_index:
        for idx in range(start_index2, rk_index +1):
            T[idx] = '0'

    # The rest of T remains as in S

    print(''.join(T))

threading.Thread(target=main).start()