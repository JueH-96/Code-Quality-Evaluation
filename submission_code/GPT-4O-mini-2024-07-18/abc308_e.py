def calculate_mex(a, b, c):
    return 3 - len(set([a, b, c]))  # MEX is 3 minus the number of unique elements

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    A = list(map(int, data[1].split()))
    S = data[2]
    
    total_sum = 0
    
    # We need to find all (i, j, k) such that S[i], S[j], S[k] = 'M', 'E', 'X'
    for i in range(N):
        if S[i] == 'M':
            for j in range(i + 1, N):
                if S[j] == 'E':
                    for k in range(j + 1, N):
                        if S[k] == 'X':
                            total_sum += calculate_mex(A[i], A[j], A[k])
    
    print(total_sum)

if __name__ == "__main__":
    main()