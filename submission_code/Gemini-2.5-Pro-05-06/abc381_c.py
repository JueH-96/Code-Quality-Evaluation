import sys

def main():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    # L1[j] = number of consecutive '1's ending at index j
    L1 = [0] * N
    if N > 0: # N is guaranteed to be >= 1 by constraints
        if S[0] == '1':
            L1[0] = 1
        for j in range(1, N):
            if S[j] == '1':
                L1[j] = L1[j-1] + 1
            # else: L1[j] is already 0 from initialization, which is correct

    # R2[j] = number of consecutive '2's starting at index j
    R2 = [0] * N
    if N > 0: # N is guaranteed to be >= 1 by constraints
        if S[N-1] == '2':
            R2[N-1] = 1
        for j in range(N-2, -1, -1): # j from N-2 down to 0
            if S[j] == '2':
                R2[j] = R2[j+1] + 1
            # else: R2[j] is already 0 from initialization, which is correct
    
    max_len = 0
    # S is guaranteed to contain at least one '/'.
    # A single '/' character is an 11/22 string of length 1 (k=0).
    # So, max_len will be at least 1 if such a character is processed.
    # Initializing with 0 is fine. When the first '/' is processed, if k=0,
    # current_len = 2*0+1 = 1, which correctly updates max_len.

    for i in range(N):
        if S[i] == '/':
            count1 = 0
            if i > 0: # Number of '1's to the left of S[i] (ending at S[i-1])
                count1 = L1[i-1]
            
            count2 = 0
            if i < N-1: # Number of '2's to the right of S[i] (starting at S[i+1])
                count2 = R2[i+1]
            
            # k is the number of '1's (and '2's) on each side of '/'
            k = min(count1, count2)
            current_len = 2 * k + 1
            
            if current_len > max_len:
                max_len = current_len
                
    sys.stdout.write(str(max_len) + "
")

if __name__ == '__main__':
    main()