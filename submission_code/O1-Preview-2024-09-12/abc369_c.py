# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    N = len(A)
    ans = N  # All single-element subarrays are arithmetic progressions
    if N == 1:
        print(ans)
        return
    curr_len = 2
    d = A[1] - A[0]
    for i in range(2, N):
        if A[i] - A[i-1] == d:
            curr_len +=1
        else:
            ans += (curr_len -1)*(curr_len -2)//2
            curr_len =2
            d = A[i] - A[i-1]
    ans += (curr_len -1)*(curr_len -2)//2
    print(ans)
threading.Thread(target=main).start()