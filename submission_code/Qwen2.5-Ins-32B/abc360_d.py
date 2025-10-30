import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    T = int(data[1])
    S = data[2]
    X = list(map(int, data[3:]))
    
    # Pair up ants that will pass each other
    pairs = 0
    pos_ants = []
    neg_ants = []
    
    for i in range(N):
        if S[i] == '1':
            pos_ants.append(X[i])
        else:
            neg_ants.append(X[i])
    
    pos_ants.sort()
    neg_ants.sort(reverse=True)
    
    for pos_ant in pos_ants:
        for neg_ant in neg_ants:
            if pos_ant < neg_ant:
                pairs += 1
            else:
                break
    
    print(pairs)

solve()