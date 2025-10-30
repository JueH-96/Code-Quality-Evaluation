# YOUR CODE HERE
import threading
import sys
import math
def main():
    import sys
    import bisect

    import sys
    sys.setrecursionlimit(1 << 25)

    N_and_T = sys.stdin.readline().strip().split()
    while len(N_and_T) < 2:
        N_and_T += sys.stdin.readline().strip().split()
    N = int(N_and_T[0])
    T = N_and_T[1]
    len_T = len(T)

    S_list = []
    total_S_length = 0
    while len(S_list) < N:
        S_line = sys.stdin.readline()
        if not S_line:
            break
        S_line = S_line.strip()
        if S_line == '':
            continue
        S_list.append(S_line)
        total_S_length += len(S_line)

    # Compute dp_prefix_len and dp_suffix_len for each S_i
    A = []  # dp_prefix_len
    B = []  # dp_suffix_len

    def match_prefix_length(S, T):
        pos_T = 0
        len_T = len(T)
        for c_S in S:
            if pos_T == len_T:
                break
            if c_S == T[pos_T]:
                pos_T +=1
        return pos_T

    def match_suffix_length(S, T):
        pos_T = len(T) -1
        for idx_S in range(len(S)-1, -1, -1):
            c_S = S[idx_S]
            if pos_T < 0:
                break
            if c_S == T[pos_T]:
                pos_T -=1
        return len(T) - pos_T -1

    for S in S_list:
        dp_prefix = match_prefix_length(S, T)
        dp_suffix = match_suffix_length(S, T)
        A.append(dp_prefix)
        B.append(dp_suffix)

    # Now compute the answer
    # For each A[i], we need to find number of B[j] such that A[i] + B[j] >= len_T

    B_sorted = sorted(B)

    total_pairs = 0
    N = len(A)
    for i in range(N):
        required_B = len_T - A[i]
        # We need B[j] >= required_B
        idx = bisect.bisect_left(B_sorted, required_B)
        count = N - idx
        total_pairs += count

    print(total_pairs)


if __name__ == '__main__':
    threading.Thread(target=main).start()