# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A_list = list(map(int, N_and_rest[1:N+1]))
    N = len(A_list)
    MAX_A_VAL = 2 * 10 ** 5 + 1
    spf = [0] * (MAX_A_VAL)
    for i in range(2, MAX_A_VAL):
        if spf[i] == 0:
            for j in range(i, MAX_A_VAL, i):
                if spf[j] == 0:
                    spf[j] = i  # smallest prime factor of j

    zero_count = 0
    s_counts = {}
    for A_i in A_list:
        if A_i == 0:
            zero_count +=1
        else:
            x = A_i
            exponents = {}
            while x >1:
                p = spf[x]
                count =0
                while x % p ==0:
                    x //= p
                    count +=1
                if count % 2 ==1:
                    exponents[p] =1
            s = 1
            for p in exponents:
                s *= p
            s_counts[s] = s_counts.get(s, 0) +1

    total_pairs = 0
    zero_pairs = zero_count*(N - zero_count) + zero_count*(zero_count -1)//2
    total_pairs += zero_pairs

    for c in s_counts.values():
        total_pairs += c*(c -1)//2

    print(total_pairs)

threading.Thread(target=main).start()