# YOUR CODE HERE
import sys

def main():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    
    left_ones = [0] * N
    for i in range(N):
        if S[i] == '1':
            if i == 0:
                left_ones[i] = 1
            else:
                left_ones[i] = left_ones[i-1] + 1
        else:
            left_ones[i] = 0
    
    right_twos = [0] * N
    for i in range(N-1, -1, -1):
        if S[i] == '2':
            if i == N-1:
                right_twos[i] = 1
            else:
                right_twos[i] = right_twos[i+1] + 1
        else:
            right_twos[i] = 0
    
    max_len = 0
    for j in range(N):
        if S[j] == '/':
            left_count = left_ones[j-1] if j > 0 else 0
            right_count = right_twos[j+1] if j+1 < N else 0
            current = 2 * min(left_count, right_count) + 1
            if current > max_len:
                max_len = current
    print(max_len)

if __name__ == "__main__":
    main()