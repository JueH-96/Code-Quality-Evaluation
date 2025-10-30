# YOUR CODE HERE
def check_condition(S, T):
    for w in range(1, len(S)):
        for c in range(1, w + 1):
            result = ''
            for i in range(0, len(S), w):
                substring = S[i:i+w]
                if len(substring) >= c:
                    result += substring[c-1]
            if result == T:
                return True
    return False

S, T = input().split()
print("Yes" if check_condition(S, T) else "No")