def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    n = int(input_data[0])
    A = list(map(int, input_data[1:]))
    
    # With two numbers, any pair forms a geometric progression.
    if n == 2:
        print("Yes")
        return
    
    # Let the common ratio be A[1]/A[0]. Instead of working with floats,
    # we use cross multiplication to avoid precision issues:
    # For a valid geometric progression, for each i (1 <= i < n),
    # A[i] should equal A[i-1] * (A[1] / A[0]), equivalently:
    # A[i] * A[0] must be equal to A[i-1] * A[1]
    r_num = A[1]  # numerator of the ratio
    r_den = A[0]  # denominator of the ratio
    
    for i in range(1, n):
        if A[i] * r_den != A[i-1] * r_num:
            print("No")
            return
    print("Yes")
    
if __name__ == '__main__':
    main()