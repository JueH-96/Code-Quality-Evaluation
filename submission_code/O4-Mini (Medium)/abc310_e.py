import sys
import threading

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1].strip()
    total = 0
    prev_sum = 0  # S_{j-1}
    # We will compute for j from 1 to N:
    # if S[j-1]=='0': S_j = j-1
    # else:            S_j = j - S_{j-1}
    for j, ch in enumerate(S, start=1):
        if ch == '0':
            curr = j - 1
        else:  # ch == '1'
            curr = j - prev_sum
        total += curr
        prev_sum = curr
    print(total)

if __name__ == "__main__":
    main()