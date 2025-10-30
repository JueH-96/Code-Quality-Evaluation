# YOUR CODE HERE
import sys

def can_be_equal(t_prime, s):
    if t_prime == s:
        return True
    if len(t_prime) == len(s) + 1:
        for i in range(len(s) + 1):
            if t_prime[:i] + t_prime[i+1:] == s:
                return True
    if len(t_prime) == len(s) - 1:
        for i in range(len(t_prime) + 1):
            if t_prime[:i] + s[i] + t_prime[i:] == s:
                return True
    if len(t_prime) == len(s):
        for i in range(len(t_prime)):
            if t_prime[:i] + s[i] + t_prime[i+1:] == s:
                return True
    return False

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    t_prime = input[1]
    S = input[2:]
    
    results = []
    for i, s in enumerate(S, start=1):
        if can_be_equal(t_prime, s):
            results.append(i)
    
    print(len(results))
    print(" ".join(map(str, results)))

if __name__ == "__main__":
    main()