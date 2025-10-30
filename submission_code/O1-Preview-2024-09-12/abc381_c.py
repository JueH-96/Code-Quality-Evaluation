# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    N = len(S)
    left_ones = [0]*N
    right_twos = [0]*N
    # Compute left_ones
    for i in range(N):
        if S[i] == '1':
            left_ones[i] = left_ones[i-1] + 1 if i > 0 else 1
        else:
            left_ones[i] = 0
    # Compute right_twos
    for i in range(N-1, -1, -1):
        if S[i] == '2':
            right_twos[i] = right_twos[i+1] + 1 if i < N-1 else 1
        else:
            right_twos[i] = 0
    max_length = 0
    for i in range(N):
        if S[i] == '/':
            left = left_ones[i-1] if i > 0 else 0
            right = right_twos[i+1] if i +1 < N else 0
            k = min(left, right)
            L = 2 * k +1
            max_length = max(max_length, L)
    print(max_length)
threading.Thread(target=main).start()