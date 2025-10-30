import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    m = int(input[1])
    A = list(map(int, input[2:2+m]))
    
    for i in range(1, n+1):
        idx = bisect.bisect_left(A, i)
        print(A[idx] - i)
        
if __name__ == "__main__":
    main()