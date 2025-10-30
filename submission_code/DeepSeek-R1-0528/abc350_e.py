from collections import deque

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = int(data[1])
    X = int(data[2])
    Y = int(data[3])
    
    if N == 0:
        print("0.000000000000000")
        return
        
    states = set()
    states.add(0)
    states.add(N)
    queue = deque([N])
    
    while queue:
        n = queue.popleft()
        if n == 0:
            continue
            
        next1 = n // A
        if next1 not in states:
            states.add(next1)
            queue.append(next1)
            
        for b in range(2, 7):
            next_b = n // b
            if next_b not in states:
                states.add(next_b)
                queue.append(next_b)
                
    sorted_states = sorted(states)
    dp = {}
    dp[0] = 0.0
    
    for n in sorted_states[1:]:
        next1 = n // A
        candidate1 = X + dp[next1]
        
        total = 0.0
        for b in range(2, 7):
            next_b = n // b
            total += dp[next_b]
        candidate2 = (6 * Y + total) / 5.0
        
        dp[n] = min(candidate1, candidate2)
        
    result = dp[N]
    print("{:.15f}".format(result))

if __name__ == "__main__":
    main()