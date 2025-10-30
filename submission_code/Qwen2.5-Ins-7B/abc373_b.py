# YOUR CODE HERE
def solve():
    S = input().strip()
    A = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    pos = {char: idx for idx, char in enumerate(S)}
    dist = 0
    current_pos = 0
    for char in A:
        dist += abs(pos[char] - current_pos)
        current_pos = pos[char]
    print(dist)
solve()