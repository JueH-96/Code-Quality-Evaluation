def main():
    import sys
    input = sys.stdin.readline
    
    # Read inputs
    N_A = input().strip().split()
    N = int(N_A[0])
    A = int(N_A[1])
    T = list(map(int, input().strip().split()))
    
    finish_time = 0
    results = []
    
    for t in T:
        # If the person arrives after the booth is free,
        # start at arrival time t, otherwise wait until finish_time.
        if t > finish_time:
            finish_time = t
        finish_time += A
        results.append(finish_time)
    
    for result in results:
        print(result)

if __name__ == '__main__':
    main()