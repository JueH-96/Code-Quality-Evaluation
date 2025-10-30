import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    T_prime = data[1]
    strings = data[2:]

    possible_indices = []

    for i in range(N):
        S = strings[i]
        if T_prime == S:
            possible_indices.append(i + 1)
        elif len(T_prime) == len(S) + 1:
            found = False
            for j in range(len(S) + 1):
                if T_prime[:j] + T_prime[j+1:] == S:
                    possible_indices.append(i + 1)
                    found = True
                    break
            if not found:
                continue
        elif len(T_prime) + 1 == len(S):
            found = False
            for j in range(len(T_prime) + 1):
                if S[:j] + S[j+1:] == T_prime:
                    possible_indices.append(i + 1)
                    found = True
                    break
            if not found:
                continue
        elif len(T_prime) == len(S):
            diff_count = 0
            for j in range(len(T_prime)):
                if T_prime[j] != S[j]:
                    diff_count += 1
                if diff_count > 1:
                    break
            if diff_count == 1:
                possible_indices.append(i + 1)

    print(len(possible_indices))
    if possible_indices:
        print(" ".join(map(str, possible_indices)))

solve()