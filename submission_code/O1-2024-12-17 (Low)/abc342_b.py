def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    # The next N values are P_1 ... P_N
    P = list(map(int, data[1:1+N]))

    Q = int(data[1+N])
    queries = data[2+N:]  # from here on, pairs of A_i, B_i

    # We'll create a dictionary to quickly find the index (position) of each person
    position = {}
    for i, person in enumerate(P):
        position[person] = i  # i is the 0-based index from the front

    idx = 0
    for _ in range(Q):
        A = int(queries[idx]); B = int(queries[idx+1])
        idx += 2
        # Compare their positions in the line
        if position[A] < position[B]:
            print(A)
        else:
            print(B)

# Do not forget to call the main function
main()