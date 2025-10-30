# YOUR CODE HERE
import sys
input = sys.stdin.read

def min_cost_to_match_or_empty(N, strings):
    results = []
    for i in range(N):
        T = strings[i]
        min_cost = len(T)  # Cost to make T empty
        for j in range(i):
            S = strings[j]
            # Calculate the minimum cost to transform T to S
            common_length = 0
            while common_length < len(T) and common_length < len(S) and T[common_length] == S[common_length]:
                common_length += 1
            cost = (len(T) - common_length) + (len(S) - common_length)
            min_cost = min(min_cost, cost)
        results.append(min_cost)
    return results

def main():
    data = input().split()
    N = int(data[0])
    strings = data[1:]
    results = min_cost_to_match_or_empty(N, strings)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()