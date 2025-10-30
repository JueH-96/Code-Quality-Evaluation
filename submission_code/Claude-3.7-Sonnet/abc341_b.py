def main():
    # Read input
    N = int(input())
    A = list(map(int, input().split()))
    S = []
    T = []
    for _ in range(N - 1):
        s, t = map(int, input().split())
        S.append(s)
        T.append(t)
    
    # Calculate the maximum number of units of country N's currency
    changed = True
    
    while changed:
        changed = False
        for i in range(N - 1):
            conversions = A[i] // S[i]
            if conversions > 0:
                A[i] -= conversions * S[i]
                A[i + 1] += conversions * T[i]
                changed = True
    
    # Print the answer
    print(A[N - 1])

if __name__ == "__main__":
    main()