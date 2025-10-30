# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    S = [sys.stdin.readline().strip() for _ in range(N)]
    prefix_min_len = {}
    total_length = 0
    for k in range(N):
        T = S[k]
        len_T = len(T)
        ans = len_T  # Cost to make T empty
        prefix = ''
        for i in range(len_T):
            prefix += T[i]
            if prefix in prefix_min_len:
                min_len_Sj = prefix_min_len[prefix]
                cost = len_T + min_len_Sj - 2*(i+1)
                if cost < ans:
                    ans = cost
        print(ans)
        # Update prefix_min_len
        prefix = ''
        for i in range(len_T):
            prefix += T[i]
            if prefix not in prefix_min_len or prefix_min_len[prefix] > len_T:
                prefix_min_len[prefix] = len_T

threading.Thread(target=main).start()