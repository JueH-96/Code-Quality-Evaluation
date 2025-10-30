# YOUR CODE HERE
import sys
import threading

def main():
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A_list = list(map(int, N_and_rest[1:N+1]))
    D = 0  # Number of discrepancies
    for i in range(N):
        initial_i = i % 2
        if A_list[i] != initial_i:
            D +=1
    MOD = 998244353
    if D % 2 == 1:
        print(0)
    else:
        ans = pow(3, D // 2, MOD)
        print(ans)

threading.Thread(target=main).start()