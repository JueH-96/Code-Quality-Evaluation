def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    A = list(map(int, input().split()))
    A.sort()
    
    # sum of elements at even 0-based indices
    fennec = sum(A[0::2])
    # sum of elements at odd  0-based indices
    snuke  = sum(A[1::2])
    
    # Fennec wins exactly when his sum > Snuke's
    if fennec > snuke:
        print("Fennec")
    else:
        print("Snuke")

if __name__ == "__main__":
    main()