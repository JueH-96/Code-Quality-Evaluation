MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    if A[0] != 0:
        print(0)
        return
    
    print(1)

if __name__ == "__main__":
    main()