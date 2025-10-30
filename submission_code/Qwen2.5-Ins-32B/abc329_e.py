# YOUR CODE HERE
def can_transform(S, T):
    N, M = len(S), len(T)
    for i in range(N - M + 1):
        if S[i:i+M] == T:
            if i == 0 or S[i-1] == '#':
                if i + M == N or S[i+M] == '#':
                    return True
    return False

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N, M = int(data[0]), int(data[1])
    S = data[2]
    T = data[3]
    
    if can_transform(S, T):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()