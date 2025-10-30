def solve():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    person_to_next = [0] * (N + 1)
    front = -1
    for i in range(1, N + 1):
        if A[i-1] == -1:
            front = i
        else:
            person_to_next[A[i-1]] = i
    result = []
    current = front
    while current != 0:
        result.append(str(current))
        current = person_to_next[current]
    print(' '.join(result))