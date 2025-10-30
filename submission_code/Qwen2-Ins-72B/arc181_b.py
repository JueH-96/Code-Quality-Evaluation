from collections import defaultdict

def preprocess(S):
    hash_values = defaultdict(int)
    current_hash = 0
    base = 257
    mod = 10**9 + 7
    base_power = 1

    for char in S:
        current_hash = (current_hash * base + ord(char)) % mod
        hash_values[current_hash] += 1
        base_power = (base_power * base) % mod

    return hash_values, base_power, base, mod

def check_patterns(S, X, Y):
    hash_values_S, base_power_S, base_S, mod_S = preprocess(S)
    hash_values_X, base_power_X, base_X, mod_X = preprocess(X)
    hash_values_Y, base_power_Y, base_Y, mod_Y = preprocess(Y)

    len_S = len(S)
    len_X = len(X)
    len_Y = len(Y)

    if len_S == 0:
        return X == Y

    if len_X != len_Y:
        return False

    current_hash_X = 0
    current_hash_Y = 0
    base_power = 1

    for i in range(len_X):
        if X[i] == '0':
            current_hash_X = (current_hash_X * base_X + ord(S[0])) % mod_X
        else:
            current_hash_X = (current_hash_X * base_X + ord(Y[i - len_S if i >= len_S else 0])) % mod_X

        if Y[i] == '0':
            current_hash_Y = (current_hash_Y * base_Y + ord(S[0])) % mod_Y
        else:
            current_hash_Y = (current_hash_Y * base_Y + ord(Y[i - len_S if i >= len_S else 0])) % mod_Y

        if i >= len_S:
            if X[i - len_S] == '0':
                current_hash_X = (current_hash_X - ord(S[0]) * base_power) % mod_X
            else:
                current_hash_X = (current_hash_X - ord(Y[i - len_S * 2 if i >= len_S * 2 else 0]) * base_power) % mod_X

            if Y[i - len_S] == '0':
                current_hash_Y = (current_hash_Y - ord(S[0]) * base_power) % mod_Y
            else:
                current_hash_Y = (current_hash_Y - ord(Y[i - len_S * 2 if i >= len_S * 2 else 0]) * base_power) % mod_Y

            base_power = (base_power * base_X) % mod_X

        if current_hash_X != current_hash_Y:
            return False

    return True

def solve(S, X, Y):
    if len(S) == 0:
        return "Yes" if X == Y else "No"

    if len(X) != len(Y):
        return "No"

    if X == Y:
        return "Yes"

    if len(X) % len(S) != 0:
        return "No"

    if not check_patterns(S, X, Y):
        return "No"

    return "Yes"

def main():
    t = int(input())
    for _ in range(t):
        S = input()
        X = input()
        Y = input()
        print(solve(S, X, Y))

if __name__ == "__main__":
    main()