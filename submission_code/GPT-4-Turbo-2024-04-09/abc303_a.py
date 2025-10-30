def are_similar_characters(x, y):
    if x == y:
        return True
    if (x == '1' and y == 'l') or (x == 'l' and y == '1'):
        return True
    if (x == '0' and y == 'o') or (x == 'o' and y == '0'):
        return True
    return False

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    T = data[2]
    
    for i in range(N):
        if not are_similar_characters(S[i], T[i]):
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()