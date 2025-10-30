def find_pair(N):
    # A simple choice for A is 2, and M can be calculated as 2^N - 1
    A = 2
    M = (1 << N) - 1  # This is 2^N - 1
    return A, M

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    T = int(data[0])
    results = []
    
    for i in range(1, T + 1):
        N = int(data[i])
        A, M = find_pair(N)
        results.append(f"{A} {M}")
    
    print("
".join(results))

if __name__ == "__main__":
    main()