# YOUR CODE HERE
import sys

def are_similar_characters(x, y):
    if x == y:
        return True
    if (x == '1' and y == 'l') or (x == 'l' and y == '1'):
        return True
    if (x == '0' and y == 'o') or (x == 'o' and y == '0'):
        return True
    return False

def are_similar_strings(S, T):
    for x, y in zip(S, T):
        if not are_similar_characters(x, y):
            return False
    return True

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    S = data[1]
    T = data[2]

    if are_similar_strings(S, T):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()