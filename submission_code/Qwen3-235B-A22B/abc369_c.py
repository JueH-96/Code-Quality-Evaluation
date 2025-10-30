import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    
    if N == 1:
        print(1)
        return
    
    # Compute the differences array
    D = []
    for i in range(N-1):
        D.append(A[i+1] - A[i])
    
    sum_run = 0
    if D:
        current = D[0]
        count = 1
        for i in range(1, len(D)):
            if D[i] == current:
                count += 1
            else:
                sum_run += count * (count + 1) // 2
                current = D[i]
                count = 1
        # Add the last run
        sum_run += count * (count + 1) // 2
    
    ans = N + sum_run
    print(ans)

if __name__ == "__main__":
    main()