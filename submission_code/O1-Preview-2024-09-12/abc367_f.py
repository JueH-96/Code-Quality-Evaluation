# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline
    N, Q = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    import random
    random.seed(0)
    N_max = max(max(A), max(B))
    H1 = [0] * (N_max + 1)
    H2 = [0] * (N_max + 1)
    H3 = [0] * (N_max + 1)
    for x in range(1, N_max + 1):
        H1[x] = random.randint(1,1<<64)
        H2[x] = random.randint(1,1<<64)
        H3[x] = random.randint(1,1<<64)

    prefix_hash_A1 = [0]*(N+1)
    prefix_hash_A2 = [0]*(N+1)
    prefix_hash_A3 = [0]*(N+1)

    prefix_hash_B1 = [0]*(N+1)
    prefix_hash_B2 = [0]*(N+1)
    prefix_hash_B3 = [0]*(N+1)

    for i in range(N):
        a_i = A[i]
        b_i = B[i]
        prefix_hash_A1[i+1] = prefix_hash_A1[i] + H1[a_i]
        prefix_hash_A2[i+1] = prefix_hash_A2[i] + H2[a_i]
        prefix_hash_A3[i+1] = prefix_hash_A3[i] + H3[a_i]

        prefix_hash_B1[i+1] = prefix_hash_B1[i] + H1[b_i]
        prefix_hash_B2[i+1] = prefix_hash_B2[i] + H2[b_i]
        prefix_hash_B3[i+1] = prefix_hash_B3[i] + H3[b_i]

    for _ in range(Q):
        l_i, r_i, L_i, R_i = map(int, sys.stdin.readline().split())
        l_i -= 1
        L_i -=1
        hA1 = prefix_hash_A1[r_i] - prefix_hash_A1[l_i]
        hA2 = prefix_hash_A2[r_i] - prefix_hash_A2[l_i]
        hA3 = prefix_hash_A3[r_i] - prefix_hash_A3[l_i]

        hB1 = prefix_hash_B1[R_i] - prefix_hash_B1[L_i]
        hB2 = prefix_hash_B2[R_i] - prefix_hash_B2[L_i]
        hB3 = prefix_hash_B3[R_i] - prefix_hash_B3[L_i]

        if hA1 == hB1 and hA2 == hB2 and hA3 == hB3:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    threading.Thread(target=main).start()