import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    people = []
    for _ in range(N):
        C_i = int(data[ptr])
        ptr += 1
        bets = set()
        for _ in range(C_i):
            bets.add(int(data[ptr]))
            ptr += 1
        people.append(bets)
    X = int(data[ptr])
    
    # Find people who have bet on X
    candidates = [i + 1 for i in range(N) if X in people[i]]
    
    if not candidates:
        print(0)
        print()
        return
    
    # Find the minimum number of bets among candidates
    min_C = min(len(people[i - 1]) for i in candidates)
    
    # Filter candidates to those with the minimum number of bets
    result = [i for i in candidates if len(people[i - 1]) == min_C]
    
    # Sort the result
    result.sort()
    
    # Output
    print(len(result))
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()