import itertools
def lexiographical_permutation(n, k, length=1, initial_sequence=None):
    if initial_sequence is None:
        initial_sequence = list(range(9, -1, -1))
    return list(itertools.islice(itertools.permutations(initial_sequence, length), n*k-1, n*k))[-1]

def smallest_321_like_number(k):
    i = 1
    while True:
        permutation = lexiographical_permutation(1, i, 10-i)
        if not permutation:
            break
        if len(permutation[-1]) == 10 - i and len(set(permutation[-1])) == 10 - i:
            if k <= 1:
                return int("".join(map(str, permutation[-1])))
            k -= 1
        i += 1

if __name__ == "__main__":
    k = int(input())
    print(smallest_321_like_number(k))