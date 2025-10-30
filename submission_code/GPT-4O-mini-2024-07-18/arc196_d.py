def can_set_strengths(N, M, Q, people, queries):
    # Create a list to track the required stamina changes
    required_stamina = [0] * (N + 1)

    # For each person, mark the required stamina changes
    for i in range(M):
        S, T = people[i]
        if S > T:
            S, T = T, S
        # The stamina at S must be 0, and at T must be 0
        # The stamina must be positive in between
        required_stamina[S] += 1
        required_stamina[T] -= 1

    # Calculate the prefix sum to determine the stamina at each town
    current_stamina = 0
    for i in range(1, N + 1):
        current_stamina += required_stamina[i]
        required_stamina[i] = current_stamina

    # Now we need to check the stamina requirements for each query
    results = []
    for L, R in queries:
        # Check the stamina requirements for the range L to R
        valid = True
        for i in range(L, R + 1):
            S, T = people[i - 1]
            if S > T:
                S, T = T, S
            # Check the stamina in the range S+1 to T-1
            for j in range(S + 1, T):
                if required_stamina[j] <= 0:
                    valid = False
                    break
            if not valid:
                break
        results.append("Yes" if valid else "No")

    return results

import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    N, M, Q = map(int, data[0].split())
    
    people = []
    for i in range(1, M + 1):
        S, T = map(int, data[i].split())
        people.append((S, T))
    
    queries = []
    for i in range(M + 1, M + 1 + Q):
        L, R = map(int, data[i].split())
        queries.append((L, R))

    results = can_set_strengths(N, M, Q, people, queries)
    print("
".join(results))

if __name__ == "__main__":
    main()