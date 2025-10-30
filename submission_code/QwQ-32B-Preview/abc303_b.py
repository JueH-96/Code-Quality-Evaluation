import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    M = int(data[ptr])
    ptr += 1
    # Generate all possible pairs
    from itertools import combinations
    people = list(range(1, N+1))
    all_pairs = list(combinations(people, 2))
    total_pairs = len(all_pairs)
    # Initialize set for adjacent pairs
    adjacent_pairs = set()
    for _ in range(M):
        # Read the photo
        photo = []
        for _ in range(N):
            photo.append(int(data[ptr]))
            ptr += 1
        # Generate adjacent pairs
        for i in range(N-1):
            pair = tuple(sorted((photo[i], photo[i+1])))
            adjacent_pairs.add(pair)
    # Calculate the answer
    answer = total_pairs - len(adjacent_pairs)
    print(answer)

if __name__ == '__main__':
    main()