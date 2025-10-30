N = int(input())
S = input()

def find_position(N, S):
    for i in range(N - 2):
        if S[i:i+3] == 'ABC':
            return i+1
    return -1

print(find_position(N, S))