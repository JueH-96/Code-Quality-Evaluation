import sys
from sys import stdin
sys.setrecursionlimit(1 << 25)

def main():
    N, M = map(int, stdin.readline().split())
    testimonies = []
    for _ in range(M):
        A, B, C = map(int, stdin.readline().split())
        A -= 1
        B -= 1
        testimonies.append((A, B, C))

    parent = list(range(N + 1))  # parent[N] is dummy node for fixed values
    parity = [0] * (N + 1)

    def find(u):
        if parent[u] != u:
            orig_parent = parent[u]
            parent[u] = find(parent[u])
            parity[u] ^= parity[orig_parent]
        return parent[u]

    S = [0] * N
    fixed = dict()

    for A, B, C in testimonies:
        rootA = find(A)
        rootB = find(B)
        current_parity = parity[A] ^ parity[B]

        if rootA != rootB:
            parent[rootB] = rootA
            parity[rootB] = current_parity ^ C
        else:
            if (current_parity ^ C) != 0:
                if B in fixed:
                    if fixed[B] != (1 - C):
                        print(-1)
                        return
                else:
                    rootB = find(B)
                    rootDummy = find(N)
                    new_parity = parity[B] ^ (1 - C)
                    if rootB != rootDummy:
                        parent[rootB] = rootDummy
                        parity[rootB] = new_parity
                        fixed[B] = 1 - C
                        S[A] = 1
                    else:
                        if (parity[B] ^ (1 - C)) != 0:
                            print(-1)
                            return
                        fixed[B] = 1 - C
                        S[A] = 1

    for B in fixed:
        rootB = find(B)
        rootDummy = find(N)
        if rootB != rootDummy:
            print(-1)
            return
        if (parity[B] ^ fixed[B]) != 0:
            print(-1)
            return

    print(''.join(map(str, S)))

main()