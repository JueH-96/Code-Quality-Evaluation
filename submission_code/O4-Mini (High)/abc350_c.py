import sys

def main():
    data = sys.stdin.readline().split()
    if not data:
        return
    n = int(data[0])
    A = [0] + list(map(int, sys.stdin.readline().split()))
    
    # pos[x] = current index of value x in A
    pos = [0] * (n + 1)
    for i in range(1, n + 1):
        pos[A[i]] = i

    ops = []
    # For each position i, if A[i] != i, swap A[i] with A[pos[i]]
    for i in range(1, n + 1):
        if A[i] != i:
            j = pos[i]            # where the value i currently is
            a_i, a_j = A[i], A[j] # backup values
            # perform swap
            A[i], A[j] = a_j, a_i
            # update positions in pos[]
            pos[a_i] = j
            pos[a_j] = i
            ops.append((i, j))

    # Output
    out = [str(len(ops))]
    for i, j in ops:
        out.append(f"{i} {j}")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()