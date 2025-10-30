def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = list(map(int, input[idx:idx+N]))
    idx += N
    A = list(map(int, input[idx:idx+N]))
    idx += N
    B = list(map(int, input[idx:idx+N]))
    idx += N

    a_max = 0
    for i in range(N):
        if A[i] > 0:
            current_max = Q[i] // A[i]
            if current_max > a_max:
                a_max = current_max

    max_total = 0

    for a in range(0, a_max + 1):
        rem = [Q[i] - a * A[i] for i in range(N)]
        min_rem = float('inf')
        valid = True
        for i in range(N):
            if B[i] > 0:
                if rem[i] < 0:
                    valid = False
                    break
                temp = rem[i] // B[i]
                if temp < min_rem:
                    min_rem = temp
        if not valid:
            continue
        if min_rem == float('inf'):
            min_rem = 0
        b = min_rem
        total = a + b
        if total > max_total:
            max_total = total

    print(max_total)

if __name__ == '__main__':
    main()