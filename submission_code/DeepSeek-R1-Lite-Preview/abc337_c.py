def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    
    behind = [0] * (N + 1)
    front_person = -1
    for i in range(1, N + 1):
        if A[i - 1] == -1:
            front_person = i
        else:
            behind[A[i - 1]] = i
    
    sequence = []
    current = front_person
    while current != 0:
        sequence.append(current)
        current = behind[current]
    
    print(' '.join(map(str, sequence)))

if __name__ == '__main__':
    main()