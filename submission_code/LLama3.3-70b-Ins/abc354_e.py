from collections import defaultdict
from sys import stdin, stdout

def read_int():
    return int(stdin.readline().strip())

def read_ints():
    return map(int, stdin.readline().strip().split())

def read_list():
    return list(map(int, stdin.readline().strip().split()))

def solve():
    n = read_int()
    cards = [read_list() for _ in range(n)]

    # Create a dictionary to store the indices of each number on the front and back sides
    front_indices = defaultdict(list)
    back_indices = defaultdict(list)
    for i, (a, b) in enumerate(cards):
        front_indices[a].append(i)
        back_indices[b].append(i)

    # Initialize a set to store the pairs of cards that can be removed
    pairs = set()
    for a in front_indices:
        if len(front_indices[a]) >= 2:
            for i in range(len(front_indices[a])):
                for j in range(i + 1, len(front_indices[a])):
                    pairs.add(tuple(sorted((front_indices[a][i], front_indices[a][j]))))
    for b in back_indices:
        if len(back_indices[b]) >= 2:
            for i in range(len(back_indices[b])):
                for j in range(i + 1, len(back_indices[b])):
                    pairs.add(tuple(sorted((back_indices[b][i], back_indices[b][j]))))

    # Initialize a dictionary to store the number of pairs that can be removed for each subset of cards
    dp = {0: 0}
    for pair in pairs:
        for mask in list(dp.keys()):
            if (mask & (1 << pair[0])) == 0 and (mask & (1 << pair[1])) == 0:
                new_mask = mask | (1 << pair[0]) | (1 << pair[1])
                dp[new_mask] = 1 - dp[mask]

    # The answer is "Takahashi" if the number of pairs that can be removed for the full subset of cards is 1, and "Aoki" otherwise
    if dp[(1 << n) - 1] == 1:
        stdout.write("Takahashi
")
    else:
        stdout.write("Aoki
")

if __name__ == "__main__":
    solve()