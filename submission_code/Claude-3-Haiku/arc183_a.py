from itertools import permutations

def get_good_sequence(N, K):
    sequences = []
    for perm in permutations(range(1, N+1), N*K):
        if all(perm[i*K:(i+1)*K] == [i+1]*K for i in range(N)):
            sequences.append(perm)
    sequences.sort()
    return sequences[((len(sequences)+1)//2)-1]

N, K = map(int, input().split())
print(" ".join(map(str, get_good_sequence(N, K))))