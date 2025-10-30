# YOUR CODE HERE
N, M = map(int, input().split())

# Initialize a list to keep track of whether family i has had their eldest son
eldest_son = [False] * (N + 1)  # index from 1 to N

# For each baby
for _ in range(M):
    Ai, Bi = input().split()
    Ai = int(Ai)
    if Bi == 'M':
        if not eldest_son[Ai]:
            # This is the eldest son in family Ai
            print('Yes')
            eldest_son[Ai] = True
        else:
            # Family Ai has already had their eldest son
            print('No')
    else:
        # Baby is female, not named Taro
        print('No')