# YOUR CODE HERE

def is_election_decided(N, T, A):
    if T > A:
        return "Yes"
    else:
        return "No"

N, T, A = map(int, input().split())
print(is_election_decided(N, T, A))