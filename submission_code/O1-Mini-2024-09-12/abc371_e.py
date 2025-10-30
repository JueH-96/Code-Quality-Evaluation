import sys

def main():
    import sys
    import sys
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    
    last_pos = [0]*(N+1)
    total = 0
    for idx in range(N):
        x = A[idx]
        prev = last_pos[x]
        contrib = (idx +1 - prev) * (N - idx)
        total += contrib
        last_pos[x] = idx +1
    print(total)

if __name__ == "__main__":
    main()