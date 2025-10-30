# YOUR CODE HERE
def roulette_bets():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    bets = []
    
    for i in range(N):
        C_i = int(data[2 * i + 1])
        A_i = list(map(int, data[2 * i + 2].split()))
        bets.append((C_i, A_i))
    
    X = int(data[2 * N + 1])
    
    # Find all persons who have bet on X
    persons = []
    min_bets = float('inf')
    
    for i in range(N):
        C_i, A_i = bets[i]
        if X in A_i:
            persons.append((i + 1, C_i))  # Store person index (1-based) and their bet count
            if C_i < min_bets:
                min_bets = C_i
    
    # Filter persons with the minimum bet count
    result = [person[0] for person in persons if person[1] == min_bets]
    
    # Print results
    print(len(result))
    if result:
        print(" ".join(map(str, sorted(result))))

roulette_bets()