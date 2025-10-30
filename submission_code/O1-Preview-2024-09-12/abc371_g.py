# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    P_list = list(map(int, N_and_rest[1:N+1]))
    A_list = list(map(int, N_and_rest[N+1:2*N+1]))

    N = int(N)
    P = [p -1 for p in P_list]  # zero-based indexing
    A = A_list  # A[0..N-1], values are from 1..N

    visited = [False] * N

    def min_rotation(s):
        n = len(s)
        ss = s + s
        i, j = 0, 1
        while i < n and j < n:
            k = 0
            while k < n and ss[i + k] == ss[j + k]:
                k += 1
            if k == n:
                break
            if ss[i + k] < ss[j + k]:
                j += k + 1
            else:
                i += k +1
            if i == j:
                j +=1
        return min(i, j)

    for idx in range(N):
        if not visited[idx]:
            positions = []
            cur_idx = idx
            while True:
                positions.append(cur_idx)
                visited[cur_idx] = True
                cur_idx = P[cur_idx]
                if visited[cur_idx]:
                    break
            values = [A[pos] for pos in positions]
            rot_start = min_rotation(values)
            rotated_values = values[rot_start:] + values[:rot_start]
            for pos, val in zip(positions, rotated_values):
                A[pos] = val

    print(' '.join(map(str, A)))

threading.Thread(target=main).start()