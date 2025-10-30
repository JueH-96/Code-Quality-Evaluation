import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    P = [0] * (N + 1)
    for i in range(1, N + 1):
        P[i] = (P[i - 1] + A[i - 1]) % M
    
    total = P[N]
    
    freq_future = defaultdict(int)
    for t in range(2, N + 1):
        freq_future[P[t]] += 1
    
    freq_past = defaultdict(int)
    answer = 0
    
    for s in range(1, N + 1):
        answer += freq_future.get(P[s], 0)
        remainder = (P[s] - total + M) % M
        answer += freq_past.get(remainder, 0)
        
        if s + 1 <= N:
            freq_future[P[s + 1]] -= 1
            if freq_future[P[s + 1]] == 0:
                del freq_future[P[s + 1]]
            freq_past[P[s + 1]] += 1
        else:
            freq_future[P[1]] -= 1
            if freq_future[P[1]] == 0:
                del freq_future[P[1]]
            freq_past[P[1]] += 1
    
    print(answer)

if __name__ == "__main__":
    main()