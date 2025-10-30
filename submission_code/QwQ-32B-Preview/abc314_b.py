import sys

def main():
    tokens = sys.stdin.read().split()
    pos = 0
    N = int(tokens[pos])
    pos += 1
    people = []
    for i in range(N):
        C_i = int(tokens[pos])
        pos += 1
        bets = set()
        for _ in range(C_i):
            num = int(tokens[pos])
            pos += 1
            bets.add(num)
        people.append({'index': i+1, 'C_i': C_i, 'bets': bets})
    X = int(tokens[pos])
    pos += 1
    candidates = [p for p in people if X in p['bets']]
    if len(candidates) == 0:
        print(0)
        print()
    else:
        min_C = min(p['C_i'] for p in candidates)
        result_indices = [p['index'] for p in candidates if p['C_i'] == min_C]
        result_indices.sort()
        print(len(result_indices))
        print(' '.join(map(str, result_indices)))

if __name__ == "__main__":
    main()