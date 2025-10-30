def main():
    import sys
    from collections import defaultdict

    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    M = int(input[ptr])
    ptr +=1
    A = list(map(int, input[ptr:ptr+N]))
    ptr +=N

    # Compute prefix_mod
    prefix_mod = [0] * (N +1)
    for i in range(1, N+1):
        prefix_mod[i] = (prefix_mod[i-1] + A[i-1]) % M
    total_mod = prefix_mod[N]

    # Compute right counts
    right_counts = [0]*N
    right_freq = defaultdict(int)
    for i in range(N-1, -1, -1):
        current = prefix_mod[i]
        right_counts[i] = right_freq.get(current, 0)
        right_freq[current] +=1

    # Compute left counts
    left_counts = [0]*N
    left_freq = defaultdict(int)
    for i in range(N):
        current = prefix_mod[i]
        target = (current - total_mod) % M
        left_counts[i] = left_freq.get(target, 0)
        left_freq[current] +=1

    # Sum all right and left counts
    total =0
    for i in range(N):
        total += right_counts[i] + left_counts[i]
    
    print(total)

if __name__ == "__main__":
    main()