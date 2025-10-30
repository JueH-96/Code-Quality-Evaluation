# YOUR CODE HERE
import sys
import threading
def main():
    import sys

    sys.setrecursionlimit(1 << 25)
    S0 = sys.stdin.readline().strip()
    len_S0 = len(S0)
    Q = int(sys.stdin.readline())
    K_list = list(map(int, sys.stdin.readline().split()))
    result = []

    for K_i in K_list:
        K = K_i
        len_S = len_S0
        # Find the minimal len_S >= K
        while len_S < K:
            len_S *= 2
        inversion = False
        while len_S > len_S0:
            half = len_S // 2
            if K <= half:
                # Remain in the left half, inversion unchanged
                pass
            else:
                # Move to right half
                K -= half
                inversion = not inversion
            len_S = half
        # Now len_S == len_S0
        char = S0[K -1]
        if inversion:
            if char.islower():
                char = char.upper()
            else:
                char = char.lower()
        result.append(char)
    print(' '.join(result))



















threading.Thread(target=main).start()