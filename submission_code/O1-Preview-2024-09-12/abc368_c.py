# YOUR CODE HERE
import sys
import threading

def main():
    import math
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    H_list = list(map(int, N_and_rest[1:]))
    T = 0
    T_mod3 = 0  # T % 3

    for H_i in H_list:
        # T_start is T +1
        T_start_mod3 = (T + 1) % 3

        # Binary search to find minimal k where D(k) >= H_i
        low = 1
        high = H_i  # Max possible number of attacks

        while low < high:
            k = (low + high) // 2

            n_cycles = k // 3
            partial_attacks = k % 3

            total_damage = n_cycles * 5 + get_partial_damage(T_start_mod3, partial_attacks)

            if total_damage >= H_i:
                high = k
            else:
                low = k + 1

        k_i = low
        T += k_i

    print(T)

def get_partial_damage(T_start_mod3, r):
    if T_start_mod3 == 0:
        if r == 0:
            return 0
        elif r == 1:
            return 3
        elif r == 2:
            return 3 + 1  # 4
    elif T_start_mod3 == 1:
        if r == 0:
            return 0
        elif r == 1:
            return 1
        elif r == 2:
            return 1 + 1  # 2
    elif T_start_mod3 == 2:
        if r == 0:
            return 0
        elif r == 1:
            return 1
        elif r == 2:
            return 1 + 3  # 4
    return 0  # Should not reach here

threading.Thread(target=main,).start()