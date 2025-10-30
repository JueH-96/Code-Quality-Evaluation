def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    Y = int(data[2])
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+2*N]))
    
    # Pair each A_i with B_i and sort by max(A_i, B_i) descending, then A_i descending
    dishes = sorted(zip(A, B), key=lambda x: (-max(x[0], x[1]), -x[0]))
    
    sum_A = 0
    sum_B = 0
    for k in range(N):
        next_A = sum_A + dishes[k][0]
        next_B = sum_B + dishes[k][1]
        # Check if adding this dish causes exceeding
        if next_A > X or next_B > Y:
            print(k + 1)
            return
        sum_A = next_A
        sum_B = next_B
    # If no dish causes exceeding, Takahashi eats all dishes
    print(N)

if __name__ == '__main__':
    main()