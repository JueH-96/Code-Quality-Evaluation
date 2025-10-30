import sys

def are_similar_characters(x, y):
    if x == y:
        return True
    if {x, y} == {'1', 'l'}:
        return True
    if {x, y} == {'0', 'o'}:
        return True
    return False

def are_similar_strings(N, S, T):
    for i in range(N):
        if not are_similar_characters(S[i], T[i]):
            return "No"
    return "Yes"

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    S = data[1]
    T = data[2]

    result = are_similar_strings(N, S, T)
    print(result)

if __name__ == "__main__":
    main()