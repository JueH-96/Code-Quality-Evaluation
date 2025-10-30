def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    P = list(map(int, input_data[1:N+1]))
    
    if N == 1:
        print(0)
    else:
        P1 = P[0]
        max_P = max(P[1:])
        x = max(0, max_P - P1 + 1)
        print(x)

if __name__ == "__main__":
    main()