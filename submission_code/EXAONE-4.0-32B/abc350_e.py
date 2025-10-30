import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    A = int(data[1])
    X = int(data[2])
    Y = int(data[3])
    
    if N == 0:
        print("{:.15f}".format(0.0))
        return
        
    states = set()
    stack = [N]
    states.add(N)
    
    while stack:
        n = stack.pop()
        if n == 0:
            continue
        next_states = set()
        n1 = n // A
        next_states.add(n1)
        for b in range(2, 7):
            nb = n // b
            next_states.add(nb)
        for s in next_states:
            if s not in states:
                states.add(s)
                stack.append(s)
                
    states.add(0)
    sorted_states = sorted(states)
    
    dp = {}
    dp[0] = 0.0
    for n in sorted_states[1:]:
        n1 = n // A
        cand1 = X + dp[n1]
        
        total = 0.0
        for b in range(2, 7):
            nb = n // b
            total += dp[nb]
            
        cand2 = (6 * Y + total) / 5.0
        
        dp[n] = min(cand1, cand2)
        
    result = dp[N]
    print("{:.15f}".format(result))

if __name__ == "__main__":
    main()