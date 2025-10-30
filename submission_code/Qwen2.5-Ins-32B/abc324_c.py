# YOUR CODE HERE
import sys

def check_strings(N, T_prime, strings):
    results = []
    for i, S in enumerate(strings, 1):
        if len(S) == len(T_prime):
            if S == T_prime or sum(c1 != c2 for c1, c2 in zip(S, T_prime)) == 1:
                results.append(i)
        elif len(S) + 1 == len(T_prime):
            for j in range(len(T_prime)):
                if S == T_prime[:j] + T_prime[j+1:]:
                    results.append(i)
                    break
        elif len(S) == len(T_prime) + 1:
            for j in range(len(S)):
                if T_prime == S[:j] + S[j+1:]:
                    results.append(i)
                    break
    return results

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    T_prime = data[1]
    strings = data[2:]
    
    results = check_strings(N, T_prime, strings)
    print(len(results))
    print(' '.join(map(str, results)))

if __name__ == "__main__":
    main()