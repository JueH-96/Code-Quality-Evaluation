from sys import stdin

def min_operations(S, T):
    """
    Find the array of strings X with the minimum number of elements obtained by 
    changing one character in S and appending S to the end of X until S equals T.

    Args:
    S (str): The initial string.
    T (str): The target string.

    Returns:
    list: The array of strings X with the minimum number of elements.
    """
    X = []
    while S != T:
        for i in range(len(S)):
            if S[i] != T[i]:
                # Change the character at position i to the corresponding character in T
                new_S = S[:i] + T[i] + S[i+1:]
                X.append(new_S)
                S = new_S
                break
    return X

def main():
    S = stdin.readline().strip()
    T = stdin.readline().strip()
    X = min_operations(S, T)
    print(len(X))
    for s in X:
        print(s)

if __name__ == "__main__":
    main()