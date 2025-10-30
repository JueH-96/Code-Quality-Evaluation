import sys

def main():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    
    if N == 0:
        print(0)
        return
    
    # Precompute prev_ones array
    prev_ones = [0] * N
    prev_ones[0] = 1 if S[0] == '1' else 0
    for i in range(1, N):
        if S[i] == '1':
            prev_ones[i] = prev_ones[i-1] + 1
        else:
            prev_ones[i] = 0
    
    # Precompute right_twos array
    right_twos = [0] * N
    right_twos[-1] = 1 if S[-1] == '2' else 0
    for i in range(N-2, -1, -1):
        if S[i] == '2':
            right_twos[i] = right_twos[i+1] + 1
        else:
            right_twos[i] = 0
    
    max_len = 0
    for m in range(N):
        if S[m] == '/':
            left = prev_ones[m-1] if m > 0 else 0
            right = right_twos[m+1] if m < N-1 else 0
            a = min(left, right)
            current = 2 * a + 1
            if current > max_len:
                max_len = current
    
    print(max_len)

if __name__ == "__main__":
    main()