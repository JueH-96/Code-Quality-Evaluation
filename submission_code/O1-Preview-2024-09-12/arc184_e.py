# YOUR CODE HERE
import sys
import threading

def main():
    import sys

    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    A = []
    for _ in range(N):
        A_i = tuple(map(int, sys.stdin.readline().split()))
        A.append(A_i)

    max_steps = 2 * M  # Maximum number of steps to simulate
    sequences = [{} for _ in range(N)]  # sequences[i][seq] = step number when seq first occurs in A_i

    for i in range(N):
        seq_seen = {}
        seq_list = []
        curr_seq = A[i]
        x = 0
        while True:
            seq_as_tuple = tuple(curr_seq)
            if seq_as_tuple in seq_seen:
                break
            seq_seen[seq_as_tuple] = x
            seq_list.append(curr_seq)
            # Apply the operation
            prefix_sum = 0
            new_seq = []
            for val in curr_seq:
                prefix_sum = (prefix_sum + val) % 2
                new_seq.append(prefix_sum)
            curr_seq = tuple(new_seq)
            x +=1
            if x > max_steps:
                break
        sequences[i] = seq_seen

    f = {}
    for i in range(N):
        seq_seen_i = sequences[i]
        for j in range(i, N):
            seq_seen_j = sequences[j]
            min_x = None
            for seq in seq_seen_i:
                if seq in seq_seen_j:
                    x_i = seq_seen_i[seq]
                    x_j = seq_seen_j[seq]
                    if x_i == x_j:
                        x = x_i  # Both sequences reach the same state at the same step
                        if min_x is None or x < min_x:
                            min_x = x
            if min_x is None:
                min_x = 0
            f_key = (i,j)
            f[f_key] = min_x

    ans = sum(f.values())
    ans %= 998244353
    print(ans)

threading.Thread(target=main).start()