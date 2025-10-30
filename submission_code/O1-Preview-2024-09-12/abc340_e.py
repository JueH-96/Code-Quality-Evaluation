# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    N_range = range(N)
    diffs = [0] * N
    total_cycles = 0
    for b in B:
        b_i = b  # Box index
        a = A[b_i]  # Number of balls in box b_i
        if a == 0:
            continue
        A[b_i] = 0  # Remove all balls from box b_i
        diffs[b_i] -= a
        full_cycles = a // N  # Number of full cycles
        remainder = a % N  # Remaining balls after full cycles
        total_cycles += full_cycles
        if remainder == 0:
            continue
        L = (b_i + 1) % N
        R_idx = (b_i + remainder) % N - 1
        if R_idx < 0:
            R_idx += N
        if L <= R_idx:
            diffs[L] += 1
            next_idx = (R_idx + 1) % N
            if next_idx != 0:
                diffs[next_idx] -=1
            else:
                diffs[0] -=1
        else:
            diffs[L] +=1
            diffs[0] +=1
            next_idx = (R_idx + 1) % N
            diffs[next_idx] -=1
    # Compute prefix sums over diffs
    net_changes = [0] * N
    cum_sum = 0
    for i in range(N):
        cum_sum += diffs[i]
        net_changes[i] = cum_sum + total_cycles
        A[i] += net_changes[i]
    print(' '.join(map(str,A)))
    

if __name__ == "__main__":
    threading.Thread(target=main).start()