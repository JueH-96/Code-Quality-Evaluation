import sys

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Check first and last elements
    if A[0] != (1 % 2) or A[-1] != (N % 2):
        print(0)
        return
    
    # Collect transitions (0-based position i means A[i] != A[i-1])
    transitions = []
    for i in range(1, N):
        if A[i] != A[i-1]:
            transitions.append(i)
    
    # If there are no transitions, only one way
    if not transitions:
        print(1)
        return
    
    # Check if number of transitions is odd (required for alternating starts and ends)
    if len(transitions) % 2 == 0:
        print(0)
        return
    
    # Precompute Fibonacci numbers up to required size
    max_fib = N + 2
    fib = [0] * max_fib
    fib[0] = 0
    fib[1] = 1
    for i in range(2, max_fib):
        fib[i] = (fib[i-1] + fib[i-2]) % MOD
    
    # Calculate product of fib[trans[j] for symmetric pairs
    res = 1
    m = len(transitions)
    for i in range((m + 1) // 2):
        left = transitions[i]
        right = transitions[m - 1 - i]
        if left >= right:
            print(0)
            return
        res = res * fib[right - left] % MOD
    
    print(res)

if __name__ == "__main__":
    main()