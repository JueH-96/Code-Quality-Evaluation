# YOUR CODE HERE
def mex(a, b, c):
    # Calculate the minimum excluded value
    if 0 not in (a, b, c):
        return 0
    if 1 not in (a, b, c):
        return 1
    if 2 not in (a, b, c):
        return 2
    return 3

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    S = data[N+1]
    
    # Precompute counts of 'E' and 'X' to the right of each position
    count_E = [0] * N
    count_X = [0] * N
    
    for i in range(N-2, -1, -1):
        count_E[i] = count_E[i+1] + (1 if S[i+1] == 'E' else 0)
        count_X[i] = count_X[i+1] + (1 if S[i+1] == 'X' else 0)
    
    # Calculate the sum of mex for all valid (i, j, k)
    total_mex_sum = 0
    for i in range(N):
        if S[i] == 'M':
            # For each 'M', find all possible 'E' and 'X' combinations
            for j in range(i+1, N):
                if S[j] == 'E':
                    # We need to find how many 'X' are there after position j
                    total_mex_sum += count_X[j] * mex(A[i], A[j], 0)
                    total_mex_sum += count_X[j] * mex(A[i], A[j], 1)
                    total_mex_sum += count_X[j] * mex(A[i], A[j], 2)
    
    print(total_mex_sum)

if __name__ == "__main__":
    main()