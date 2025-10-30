def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    Q = int(data[N+1])
    queries = data[N+2:]

    # Map each person's number to its position in the line (0-based index)
    position = {}
    for i, person in enumerate(P):
        position[person] = i

    index = 0
    for _ in range(Q):
        A = int(queries[index])
        B = int(queries[index + 1])
        index += 2

        # The person with the smaller index is further to the front
        if position[A] < position[B]:
            print(A)
        else:
            print(B)

def main():
    solve()

if __name__ == "__main__":
    main()