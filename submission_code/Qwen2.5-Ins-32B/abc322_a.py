# YOUR CODE HERE
def find_abc_position():
    N = int(input())
    S = input().strip()
    
    for i in range(N - 2):
        if S[i:i+3] == "ABC":
            return i + 1
    return -1

print(find_abc_position())