import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    if N == 0:
        print(0)
        return
    total = N
    if N >=2:
        total += (N-1)
    if N <3:
        print(total)
        return
    run_length =2
    for i in range(2,N):
        if A[i] - A[i-1] == A[i-1] - A[i-2]:
            run_length +=1
        else:
            if run_length >=3:
                total += (run_length -2)*(run_length -1)//2
            run_length =2
    if run_length >=3:
        total += (run_length -2)*(run_length -1)//2
    print(total)

if __name__ == "__main__":
    main()