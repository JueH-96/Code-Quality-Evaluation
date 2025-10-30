def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # positions[i] will store the indices where the value i appears in A
    positions = [[] for _ in range(N+1)]

    # Record indices (1-based) of each value in A
    for idx, val in enumerate(A, 1):
        positions[val].append(idx)

    # f_list will store tuples of the form (f(i), i), where f(i) is the middle index of i
    f_list = []
    for i in range(1, N+1):
        # positions[i] has exactly 3 indices: alpha < beta < gamma
        # f(i) = beta (the middle index)
        alpha, beta, gamma = positions[i]
        f_list.append((beta, i))

    # Sort by the middle index f(i)
    f_list.sort(key=lambda x: x[0])

    # Output the values in ascending order of their middle indices
    print(" ".join(str(elem[1]) for elem in f_list))

def main():
    solve()

if __name__ == "__main__":
    main()