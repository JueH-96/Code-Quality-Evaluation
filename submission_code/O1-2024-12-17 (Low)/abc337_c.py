def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Find the person at the front (A_i = -1) 
    front = -1
    # We'll store for each person X who stands behind them
    behind = [0]*(N+1)  # 1-based indexing, 0 means no one behind

    for i in range(1, N+1):
        if A[i-1] == -1:
            front = i
        else:
            behind[A[i-1]] = i

    # Reconstruct the order
    order = []
    current = front
    while current != 0:
        order.append(str(current))
        current = behind[current]

    # Print the result
    print(" ".join(order))

# Do not forget to call main
main()