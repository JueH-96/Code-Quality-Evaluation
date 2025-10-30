# YOUR CODE HERE
import sys
from bisect import bisect_left, bisect_right

def main():
    import threading
    def solve():
        import sys
        N_and_rest = sys.stdin.read().split()
        N = int(N_and_rest[0])
        A = list(map(int, N_and_rest[1:N+1]))
        max_A = max(A)
        freq = [0] * (max_A+1)
        for a in A:
            freq[a] +=1
        
        total = 0
        max_Mult = max_A
        
        for s in range(1, max_A+1):
            if freq[s]==0:
                continue
            for mult in range(1, max_A//s +1):
                l = s * mult
                if l < s:
                    continue
                if l > max_A:
                    break
                if freq[l]==0:
                    continue
                k = mult
                if s == l:
                    num_pairs = freq[s] * (freq[s] -1) // 2
                else:
                    num_pairs = freq[s] * freq[l]
                total += num_pairs * k
        print(total)
    threading.Thread(target=solve).start()