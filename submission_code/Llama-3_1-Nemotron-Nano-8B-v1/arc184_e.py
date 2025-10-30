import sys
MOD = 998244353

def main():
    N, M = map(int, sys.stdin.readline().split())
    sequences = []
    for _ in range(N):
        seq = list(map(int, sys.stdin.readline().split()))
        sequences.append(tuple(seq))
    
    total = 0
    for i in range(N):
        a = sequences[i]
        a_dict = {}
        a_steps = []
        for x in range(1000):  # Limit steps to avoid infinite loops
            if a in a_dict:
                break
            a_dict[a] = x
            a_steps.append(a)
            new_a = []
            for k in range(M):
                s = sum(a[:k+1]) % 2
                new_a.append(s)
            a = tuple(new_a)
        for j in range(i + 1, N):
            b = sequences[j]
            if a == b:
                total = (total + 0) % MOD
                continue
            b_dict = {}
            b_steps = []
            found = False
            x = 0
            current_a = a
            current_b = b
            for x in range(1000):  # Limit steps to avoid infinite loops
                if current_a == current_b:
                    total = (total + x) % MOD
                    found = True
                    break
                if current_a in a_dict:
                    break
                if current_b in b_dict:
                    break
                b_dict[current_b] = x
                b_steps.append(current_b)
                new_a = []
                for k in range(M):
                    s = sum(current_a[:k+1]) % 2
                    new_a.append(s)
                current_a = tuple(new_a)
                new_b = []
                for k in range(M):
                    s = sum(current_b[:k+1]) % 2
                    new_b.append(s)
                current_b = tuple(new_b)
            if not found:
                continue
    print(total % MOD)

if __name__ == '__main__':
    main()