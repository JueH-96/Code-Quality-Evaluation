def find_winner():
    N = int(input().strip())
    A = list(map(int, input().strip().split()))

    # If N is even, Anna can always win. If N is odd, Bruno can always win if the greatest common divisor of all numbers in A is 1.
    gcd = A[0]
    for i in range(1, N):
        gcd = math.gcd(gcd, A[i])

    if gcd == 1:
        print("Bruno")
    else:
        print("Anna")

find_winner()