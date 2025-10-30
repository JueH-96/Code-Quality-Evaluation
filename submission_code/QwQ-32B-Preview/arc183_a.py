import sys
from functools import lru_cache

def main():
    sys.setrecursionlimit(1 << 25)
    N_K = sys.stdin.read().split()
    N = int(N_K[0])
    K = int(N_K[1])
    
    @lru_cache(maxsize=None)
    def sequence_count(pos, counts_tuple):
        if pos == N * K:
            return 1
        counts = list(counts_tuple)
        total = 0
        for i in range(N):
            if counts[i] < K:
                counts[i] += 1
                total += sequence_count(pos + 1, tuple(counts))
                counts[i] -= 1
        return total
    
    sequence = []
    counts = [0] * N
    M = (sequence_count(0, tuple(counts)) + 1) // 2
    remaining = M
    for pos in range(N * K):
        for i in range(N):
            if counts[i] < K:
                counts[i] += 1
                cnt = sequence_count(pos + 1, tuple(counts))
                counts[i] -= 1
                if remaining <= cnt:
                    sequence.append(i + 1)
                    remaining -= 1
                    break
                else:
                    remaining -= cnt
                    sequence.append(i + 1)
                    counts[i] += 1
        else:
            break
    print(' '.join(map(str, sequence)))

if __name__ == '__main__':
    main()